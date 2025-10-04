#!/usr/bin/env python3
"""
Glucose Data Preprocessor for Machine Learning

This script processes glucose monitoring data for ML training by:
1. Detecting time gaps in the data
2. Interpolating missing values for gaps <= 10 minutes
3. Creating sequence IDs for continuous data segments
4. Creating fixed-frequency data with consistent intervals
5. Providing statistics about the processed data
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import csv
import warnings
import yaml
import sys

# Import database detection and conversion classes
from formats import DatabaseDetector

warnings.filterwarnings('ignore')

# Set console encoding to UTF-8 to handle Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


class GlucoseMLPreprocessor:
    """
    Preprocessor for glucose monitoring data to prepare it for machine learning.
    
    This preprocessor performs the following steps:
    1. Consolidates multiple CSV files
    2. Replaces High/Low glucose values with numeric equivalents
    3. Removes calibration events
    4. Detects gaps and creates sequence IDs
    5. Interpolates missing values for small gaps
    6. Filters sequences by minimum length
    7. Creates fixed-frequency data with consistent intervals (optional)
    8. Optionally filters to glucose-only data
    9. Prepares final ML-ready dataset
    """
    
    @classmethod
    def from_config_file(cls, config_path: str, **cli_overrides):
        """
        Create a GlucoseMLPreprocessor instance from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            **cli_overrides: Command line arguments that override config values
            
        Returns:
            GlucoseMLPreprocessor instance with loaded configuration
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Extract database-specific configurations
        dexcom_config = config.get('dexcom', {})
        high_value = dexcom_config.get('high_glucose_value', 401)
        low_value = dexcom_config.get('low_glucose_value', 39)
        
        # Create instance with config values, overridden by CLI arguments
        return cls(
            expected_interval_minutes=cli_overrides.get('expected_interval_minutes', config.get('expected_interval_minutes', 5)),
            small_gap_max_minutes=cli_overrides.get('small_gap_max_minutes', config.get('small_gap_max_minutes', 15)),
            remove_calibration=cli_overrides.get('remove_calibration', dexcom_config.get('remove_calibration', True)),
            min_sequence_len=cli_overrides.get('min_sequence_len', config.get('min_sequence_len', 200)),
            save_intermediate_files=cli_overrides.get('save_intermediate_files', config.get('save_intermediate_files', False)),
            calibration_period_minutes=cli_overrides.get('calibration_period_minutes', dexcom_config.get('calibration_period_minutes', 165)),
            remove_after_calibration_hours=cli_overrides.get('remove_after_calibration_hours', dexcom_config.get('remove_after_calibration_hours', 24)),
            high_glucose_value=cli_overrides.get('high_glucose_value', high_value),
            low_glucose_value=cli_overrides.get('low_glucose_value', low_value),
            glucose_only=cli_overrides.get('glucose_only', config.get('glucose_only', False)),
            create_fixed_frequency=cli_overrides.get('create_fixed_frequency', config.get('create_fixed_frequency', True)),
            config=config
        )
    
    def __init__(self, expected_interval_minutes: int = 5, small_gap_max_minutes: int = 15, remove_calibration: bool = True, min_sequence_len: int = 200, save_intermediate_files: bool = False, calibration_period_minutes: int = 60*2 + 45, remove_after_calibration_hours: int = 24, high_glucose_value: int = 401, low_glucose_value: int = 39, glucose_only: bool = False, create_fixed_frequency: bool = True, config: Optional[Dict] = None):
        """
        Initialize the preprocessor.
        
        Args:
            expected_interval_minutes: Expected data collection interval for time discretization (default: 5 minutes)
            small_gap_max_minutes: Maximum gap size to interpolate (default: 15 minutes)
            remove_calibration: If True, remove all Calibration Event Type rows (default: True)
            min_sequence_len: Minimum sequence length to keep for ML training (default: 200)
            save_intermediate_files: If True, save intermediate files after each processing step (default: False)
            calibration_period_minutes: Gap duration considered as calibration period (default: 165 minutes)
            remove_after_calibration_hours: Hours of data to remove after calibration period (default: 24 hours)
            high_glucose_value: Numeric value to replace 'High' glucose readings (default: 401 mg/dL)
            low_glucose_value: Numeric value to replace 'Low' glucose readings (default: 39 mg/dL)
            glucose_only: If True, output only glucose data with simplified fields (default: False)
            create_fixed_frequency: If True, create fixed-frequency data with consistent intervals (default: True)
            config: Optional configuration dictionary from YAML file
        """
        self.expected_interval_minutes = expected_interval_minutes
        self.small_gap_max_minutes = small_gap_max_minutes
        self.remove_calibration = remove_calibration
        self.min_sequence_len = min_sequence_len
        self.save_intermediate_files = save_intermediate_files
        self.calibration_period_minutes = calibration_period_minutes
        self.remove_after_calibration_hours = remove_after_calibration_hours
        self.high_glucose_value = high_glucose_value
        self.low_glucose_value = low_glucose_value
        self.glucose_only = glucose_only
        self.create_fixed_frequency = create_fixed_frequency
        self.config = config
        self.expected_interval_seconds = expected_interval_minutes * 60
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        self.calibration_period_seconds = calibration_period_minutes * 60
    
    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object for sorting."""
        if not timestamp_str or timestamp_str.strip() == "":
            return None
        
        # Handle the format "2019-10-28 0:52:15" or "2019-10-14T16:42:37"
        timestamp_str = timestamp_str.strip()
        
        # Try different timestamp formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",  # ISO format with T
            "%Y-%m-%d %H:%M:%S",  # Space format
            "%Y-%m-%d %H:%M:%S.%f",  # With microseconds
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
    
    
    def consolidate_glucose_data(self, data_folder: str, output_file: str = None) -> pl.DataFrame:
        """Consolidate all data files in the folder into a single DataFrame.
        
        Args:
            data_folder: Path to folder containing data files
            output_file: Optional path to save consolidated data
            
        Returns:
            DataFrame with consolidated and sorted data
        """
        # Detect database type and get appropriate converter
        db_detector = DatabaseDetector()
        database_type = db_detector.detect_database_type(data_folder)
        
        print(f"Detected database type: {database_type}")
        
        if database_type == 'unknown':
            raise ValueError(f"Could not detect database type for folder: {data_folder}")
        
        # Get database converter
        database_converter = db_detector.get_database_converter(database_type, self.config or {})
        
        if database_converter is None:
            raise ValueError(f"No converter available for database type: {database_type}")
        
        print(f"Using {database_converter.get_database_name()}")
        
        # Consolidate data using the appropriate converter
        df = database_converter.consolidate_data(data_folder, output_file)
        
        # Store original record count for statistics
        self._original_record_count = len(df)

        return df
        
    
    def detect_gaps_and_sequences(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Detect time gaps and create sequence IDs, marking calibration periods and sequences for removal.
        
        Args:
            df: DataFrame with timestamp column and optionally user_id column
            
        Returns:
            Tuple of (DataFrame with sequence IDs and removal flags, statistics dictionary)
        """
        print("Detecting gaps and creating sequences...")
        
        # Handle multi-user data by processing each user separately
        if 'user_id' in df.columns:
            print("Processing multi-user data - creating sequences per user...")
            all_sequences = []
            
            for user_id in df['user_id'].unique():
                user_data = df.filter(pl.col('user_id') == user_id).sort('timestamp')
                user_sequences = self._create_sequences_for_user(user_data, user_id)
                all_sequences.append(user_sequences)
            
            # Combine all user sequences
            df = pl.concat(all_sequences).sort(['user_id', 'sequence_id', 'timestamp'])
        else:
            # Single user data - process normally
            df = df.sort('timestamp')
            df = self._create_sequences_for_user(df)
        
        # Calculate statistics
        sequence_counts = df.group_by(['user_id', 'sequence_id'] if 'user_id' in df.columns else ['sequence_id']).count().sort(['user_id', 'sequence_id'] if 'user_id' in df.columns else 'sequence_id')
        stats = {
            'total_sequences': df['sequence_id'].max() + 1,
            'gap_positions': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
            'total_gaps': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
            'sequence_lengths': dict(zip(sequence_counts['sequence_id'].to_list(), sequence_counts['count'].to_list())),
            'calibration_period_analysis': {'calibration_periods_detected': 0, 'sequences_marked_for_removal': 0, 'total_records_marked_for_removal': 0}
        }
        
        print(f"Created {stats['total_sequences']} sequences")
        print(f"Found {stats['total_gaps']} gaps > {self.small_gap_max_minutes} minutes")
        
        # Remove temporary columns
        columns_to_remove = ['time_diff_seconds', 'is_gap']
        df = df.drop([col for col in columns_to_remove if col in df.columns])
        
        return df, stats
    
    def _create_sequences_for_user(self, user_df: pl.DataFrame, user_id: str = None) -> pl.DataFrame:
        """
        Create sequences for a single user's data.
        
        Args:
            user_df: DataFrame with data for a single user
            user_id: User ID (for multi-user data)
            
        Returns:
            DataFrame with sequence IDs added
        """
        # Calculate time differences and create sequence IDs
        df = user_df.with_columns([
                    pl.col('timestamp').diff().dt.total_seconds().alias('time_diff_seconds'),
                ]).with_columns([
                    (pl.col('time_diff_seconds') > self.small_gap_max_seconds).alias('is_gap'),
                ]).with_columns([
                    pl.col('is_gap').cum_sum().alias('sequence_id')
                ])
        
        # For multi-user data, create unique sequence IDs by combining user_id and sequence_id
        if user_id is not None:
            # Create global sequence IDs by offsetting based on user
            max_previous_sequences = 0  # This would need to be tracked across users
            # For now, we'll use a simpler approach: user_id * 100000 + sequence_id
            df = df.with_columns([
                (pl.lit(int(user_id) * 100000) + pl.col('sequence_id')).alias('sequence_id')
            ])
        
        return df
    
    def interpolate_missing_values(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Interpolate only small gaps (1-2 missing data points) within sequences.
        Large gaps are treated as sequence boundaries and not interpolated.
        
        Args:
            df: DataFrame with sequence IDs and timestamp data
            
        Returns:
            Tuple of (DataFrame with interpolated values, interpolation statistics)
        """
        print("Interpolating small gaps only...")
        
        interpolation_stats = {
            'total_interpolations': 0,
            'total_interpolated_data_points': 0,
            'glucose_value_mg/dl_interpolations': 0,
            'insulin_value_u_interpolations': 0,
            'carb_value_grams_interpolations': 0,
            'sequences_processed': 0,
            'small_gaps_filled': 0,
            'large_gaps_skipped': 0
        }
        
        # Process each sequence separately
        unique_sequences = df['sequence_id'].unique().to_list()
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col('sequence_id') == seq_id).sort('timestamp')
            
            if len(seq_data) < 2:
                continue
                
            interpolation_stats['sequences_processed'] += 1
            
            # Get time differences as list for processing
            time_diffs = seq_data['timestamp'].diff().dt.total_seconds() / 60.0
            time_diffs_list = time_diffs.to_list()
            
            # Find small gaps (1-2 missing data points = expected_interval to small_gap_max_minutes)
            small_gaps = [(i, diff) for i, diff in enumerate(time_diffs_list) 
                         if i > 0 and self.expected_interval_minutes < diff <= self.small_gap_max_minutes]
            large_gaps = [(i, diff) for i, diff in enumerate(time_diffs_list) 
                         if i > 0 and diff > self.small_gap_max_minutes]
            
            interpolation_stats['small_gaps_filled'] += len(small_gaps)
            interpolation_stats['large_gaps_skipped'] += len(large_gaps)
            
            # Only interpolate small gaps
            if small_gaps:
                # Convert to pandas for easier interpolation logic, then back to polars
                seq_pandas = seq_data.to_pandas()
                new_rows = []
                
                for gap_idx, time_diff_minutes in small_gaps:
                    if gap_idx > 0:
                        prev_row = seq_pandas.iloc[gap_idx-1]
                        current_row = seq_pandas.iloc[gap_idx]
                        
                        # Calculate number of missing points
                        missing_points = int(time_diff_minutes / self.expected_interval_minutes) - 1
                        
                        if missing_points > 0:
                            # Create interpolated points
                            for j in range(1, missing_points + 1):
                                interpolated_time = prev_row['timestamp'] + timedelta(minutes=self.expected_interval_minutes*j)
                                
                                # Interpolate numeric values - include all columns from original data
                                new_row = {
                                    'Timestamp (YYYY-MM-DDThh:mm:ss)': interpolated_time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    'Event Type': 'Interpolated',
                                    'Glucose Value (mg/dL)': None,  # Default to None (will be converted to proper type)
                                    'Insulin Value (u)': None,      # Default to None (will be converted to proper type)
                                    'Carb Value (grams)': None,     # Default to None (will be converted to proper type)
                                    'timestamp': interpolated_time,
                                    'sequence_id': seq_id
                                }
                                
                                # Add user_id if it exists in the original data
                                if 'user_id' in seq_pandas.columns:
                                    new_row['user_id'] = prev_row['user_id']
                                
                                # Linear interpolation for numeric columns
                                numeric_cols = ['Glucose Value (mg/dL)', 'Insulin Value (u)', 'Carb Value (grams)']
                                interpolations_made = 0
                                for col in numeric_cols:
                                    prev_val = prev_row[col]
                                    curr_val = current_row[col]
                                    
                                    # Check if both values are valid and numeric
                                    try:
                                        prev_numeric = float(prev_val) if prev_val is not None and str(prev_val).strip() != '' else None
                                        curr_numeric = float(curr_val) if curr_val is not None and str(curr_val).strip() != '' else None
                                        
                                        if prev_numeric is not None and curr_numeric is not None:
                                            # Linear interpolation
                                            alpha = j / (missing_points + 1)
                                            interpolated_value = prev_numeric + alpha * (curr_numeric - prev_numeric)
                                            new_row[col] = interpolated_value  # Keep as numeric value
                                            interpolation_stats[f'{col.lower().replace(" ", "_").replace("(", "").replace(")", "")}_interpolations'] += 1
                                            interpolations_made += 1
                                    except (ValueError, TypeError):
                                        # Keep empty string for non-numeric values
                                        pass
                                
                                # Count this as one interpolated data point if any field was interpolated
                                if interpolations_made > 0:
                                    interpolation_stats['total_interpolations'] += 1
                                
                                new_rows.append(new_row)
                                
                                # Count this as one interpolated data point (row created)
                                interpolation_stats['total_interpolated_data_points'] += 1
                
                # Add interpolated rows to the sequence
                if new_rows:
                    # Create DataFrame with explicit schema based on original sequence
                    schema = {col: seq_data[col].dtype for col in seq_data.columns}
                    interpolated_df = pl.DataFrame(new_rows, schema=schema)
                    # Ensure column order matches original sequence
                    interpolated_df = interpolated_df.select(seq_data.columns)
                    # Sort by user_id and timestamp if user_id exists, otherwise just by sequence_id and timestamp
                    if 'user_id' in seq_data.columns:
                        seq_data = pl.concat([seq_data, interpolated_df]).sort(['user_id', 'sequence_id', 'timestamp'])
                    else:
                        seq_data = pl.concat([seq_data, interpolated_df]).sort(['sequence_id', 'timestamp'])
            
            # Update the main DataFrame
            df = df.filter(pl.col('sequence_id') != seq_id)  # Remove original sequence data
            df = pl.concat([df, seq_data])
        
        # Sort by user_id, sequence_id and timestamp if user_id exists, otherwise just by sequence_id and timestamp
        if 'user_id' in df.columns:
            df = df.sort(['user_id', 'sequence_id', 'timestamp'])
        else:
            df = df.sort(['sequence_id', 'timestamp'])
        
        print(f"Identified and processed {interpolation_stats['small_gaps_filled']} small gaps")
        print(f"Created {interpolation_stats['total_interpolated_data_points']} interpolated data points")
        print(f"Interpolated {interpolation_stats['total_interpolations']} missing field values")
        print(f"Skipped {interpolation_stats['large_gaps_skipped']} large gaps")
        print(f"Processed {interpolation_stats['sequences_processed']} sequences")
        
        return df, interpolation_stats
    
    
    def filter_sequences_by_length(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filter out sequences that are shorter than the minimum required length.
        
        Args:
            df: DataFrame with sequence IDs and processed data
            
        Returns:
            Tuple of (filtered DataFrame, filtering statistics dictionary)
        """
        print(f"Filtering sequences with length < {self.min_sequence_len}...")
        
        # Calculate sequence lengths
        sequence_counts = df.group_by('sequence_id').count().sort('sequence_id')
        
        # Find sequences to keep (longer than or equal to min_sequence_len)
        sequences_to_keep = sequence_counts.filter(pl.col('count') >= self.min_sequence_len)
        
        filtering_stats = {
            'original_sequences': sequence_counts.height,
            'filtered_sequences': sequences_to_keep.height,
            'removed_sequences': sequence_counts.height - sequences_to_keep.height,
            'original_records': len(df),
            'filtered_records': 0,  # Will be calculated after filtering
            'removed_records': 0    # Will be calculated after filtering
        }
        
        if len(sequences_to_keep) == 0:
            print("Warning: No sequences meet the minimum length requirement!")
            return df, filtering_stats
        
        # Filter the DataFrame to keep only sequences that meet the length requirement
        valid_sequence_ids = sequences_to_keep['sequence_id'].to_list()
        filtered_df = df.filter(pl.col('sequence_id').is_in(valid_sequence_ids))
        
        # Update filtering statistics
        filtering_stats['filtered_records'] = len(filtered_df)
        filtering_stats['removed_records'] = len(df) - len(filtered_df)
        
        print(f"Original sequences: {filtering_stats['original_sequences']}")
        print(f"Sequences after filtering: {filtering_stats['filtered_sequences']}")
        print(f"Sequences removed: {filtering_stats['removed_sequences']}")
        print(f"Original records: {filtering_stats['original_records']:,}")
        print(f"Records after filtering: {filtering_stats['filtered_records']:,}")
        print(f"Records removed: {filtering_stats['removed_records']:,}")
        
        # Show statistics about removed sequences
        if filtering_stats['removed_sequences'] > 0:
            removed_sequences = sequence_counts.filter(pl.col('count') < self.min_sequence_len)
            if len(removed_sequences) > 0:
                min_len_removed = removed_sequences['count'].min()
                max_len_removed = removed_sequences['count'].max()
                avg_len_removed = removed_sequences['count'].mean()
                print(f"Removed sequence lengths - Min: {min_len_removed}, Max: {max_len_removed}, Avg: {avg_len_removed:.1f}")
        
        return filtered_df, filtering_stats
    
    def create_fixed_frequency_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Create fixed-frequency data by aligning sequences to round minutes and ensuring consistent intervals.
        Glucose values are interpolated, while carbs and insulin are shifted to closest datapoints.
        
        Args:
            df: DataFrame with processed data and sequence IDs
            
        Returns:
            Tuple of (DataFrame with fixed-frequency data, statistics dictionary)
        """
        print(f"Creating fixed-frequency data with {self.expected_interval_minutes}-minute intervals...")
        
        fixed_freq_stats = {
            'sequences_processed': 0,
            'total_records_before': len(df),
            'total_records_after': 0,
            'glucose_interpolations': 0,
            'carb_shifted_records': 0,
            'insulin_shifted_records': 0,
            'time_adjustments': 0
        }
        
        # Process each sequence separately
        unique_sequences = df['sequence_id'].unique().to_list()
        all_fixed_sequences = []
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col('sequence_id') == seq_id).sort('timestamp')
            
            if len(seq_data) < 2:
                # Keep single-point sequences as-is
                all_fixed_sequences.append(seq_data)
                continue
                
            fixed_freq_stats['sequences_processed'] += 1
            
            # Convert to pandas for easier time manipulation
            seq_pandas = seq_data.to_pandas()
            
            # Get the first timestamp and align it to the nearest round minute
            first_timestamp = seq_pandas['timestamp'].iloc[0]
            first_second = first_timestamp.second
            
            # Calculate adjustment needed to align to round minute
            if first_second >= 30:
                # Round up to next minute
                adjustment_seconds = 60 - first_second
            else:
                # Round down to current minute
                adjustment_seconds = -first_second
            
            # Create aligned start time
            aligned_start = first_timestamp + timedelta(seconds=adjustment_seconds)
            if adjustment_seconds != 0:
                fixed_freq_stats['time_adjustments'] += 1
            
            # Calculate the end time of the sequence
            last_timestamp = seq_pandas['timestamp'].iloc[-1]
            
            # Create fixed-frequency timestamps
            current_time = aligned_start
            fixed_timestamps = []
            while current_time <= last_timestamp:
                fixed_timestamps.append(current_time)
                current_time += timedelta(minutes=self.expected_interval_minutes)
            
            # Create new DataFrame with fixed timestamps
            fixed_rows = []
            
            for fixed_time in fixed_timestamps:
                # Find the closest original timestamp
                time_diffs = abs((seq_pandas['timestamp'] - fixed_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                      
                # Create new row
                new_row = {
                    'Timestamp (YYYY-MM-DDThh:mm:ss)': fixed_time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'timestamp': fixed_time,
                    'sequence_id': seq_id
                }
                
                # Handle glucose interpolation - always interpolate glucose values
                if 'Glucose Value (mg/dL)' in seq_pandas.columns:
                    # Find valid glucose values for interpolation
                    valid_glucose_mask = (
                        seq_pandas['Glucose Value (mg/dL)'].notna() & 
                        (seq_pandas['Glucose Value (mg/dL)'] != '') &
                        (seq_pandas['Glucose Value (mg/dL)'].astype(str).str.strip() != '')
                    )
                    
                    if valid_glucose_mask.any():
                        # Get valid glucose points
                        valid_glucose_data = seq_pandas[valid_glucose_mask].copy()
                        
                        # Calculate time differences to find closest valid glucose points
                        time_diffs = abs((valid_glucose_data['timestamp'] - fixed_time).dt.total_seconds())
                        sorted_indices = time_diffs.argsort()
                        
                        if len(sorted_indices) >= 2:
                            # Get two closest valid glucose points
                            idx1 = sorted_indices.iloc[0]
                            idx2 = sorted_indices.iloc[1]
                            point1 = valid_glucose_data.iloc[idx1]
                            point2 = valid_glucose_data.iloc[idx2]
                            
                            # Linear interpolation between valid glucose points
                            try:
                                glucose1 = float(point1['Glucose Value (mg/dL)'])
                                glucose2 = float(point2['Glucose Value (mg/dL)'])
                                
                                # Calculate interpolation weight
                                total_time = abs((point2['timestamp'] - point1['timestamp']).total_seconds())
                                if total_time > 0:
                                    weight1 = abs((point2['timestamp'] - fixed_time).total_seconds()) / total_time
                                    weight2 = abs((fixed_time - point1['timestamp']).total_seconds()) / total_time
                                    
                                    interpolated_glucose = (weight1 * glucose1 + weight2 * glucose2)
                                    new_row['Glucose Value (mg/dL)'] = interpolated_glucose
                                    fixed_freq_stats['glucose_interpolations'] += 1
                                else:
                                    new_row['Glucose Value (mg/dL)'] = glucose1
                            except (ValueError, TypeError):
                                # If conversion fails, use closest valid glucose value
                                closest_valid_idx = valid_glucose_data.iloc[sorted_indices.iloc[0]]
                                new_row['Glucose Value (mg/dL)'] = closest_valid_idx['Glucose Value (mg/dL)']
                        else:
                            # Use closest valid glucose value if not enough points for interpolation
                            closest_valid_idx = valid_glucose_data.iloc[sorted_indices.iloc[0]]
                            new_row['Glucose Value (mg/dL)'] = closest_valid_idx['Glucose Value (mg/dL)']
                    else:
                        # No valid glucose values found - this shouldn't happen in practice
                        # but we'll set to None to avoid errors
                        new_row['Glucose Value (mg/dL)'] = None
                
                # Handle insulin and carb shifting (use closest value)
                if 'Insulin Value (u)' in seq_pandas.columns:
                    new_row['Insulin Value (u)'] = seq_pandas['Insulin Value (u)'].iloc[closest_idx]
                    if seq_pandas['Insulin Value (u)'].iloc[closest_idx] is not None:
                        fixed_freq_stats['insulin_shifted_records'] += 1
                
                if 'Carb Value (grams)' in seq_pandas.columns:
                    new_row['Carb Value (grams)'] = seq_pandas['Carb Value (grams)'].iloc[closest_idx]
                    if seq_pandas['Carb Value (grams)'].iloc[closest_idx] is not None:
                        fixed_freq_stats['carb_shifted_records'] += 1
                
                # Copy other fields from closest record (but not the timestamp string)
                for col in seq_pandas.columns:
                    if col not in ['timestamp', 'sequence_id', 'Glucose Value (mg/dL)', 'Insulin Value (u)', 'Carb Value (grams)', 'Timestamp (YYYY-MM-DDThh:mm:ss)']:
                        new_row[col] = seq_pandas[col].iloc[closest_idx]
                
                fixed_rows.append(new_row)
            
            # Convert back to polars DataFrame
            if fixed_rows:
                # Create DataFrame with explicit schema based on original sequence
                schema = {col: seq_data[col].dtype for col in seq_data.columns}
                fixed_seq_df = pl.DataFrame(fixed_rows, schema=schema)
                all_fixed_sequences.append(fixed_seq_df)
        
        # Combine all fixed sequences
        if all_fixed_sequences:
            df_fixed = pl.concat(all_fixed_sequences).sort(['sequence_id', 'timestamp'])
        else:
            df_fixed = df
        
        fixed_freq_stats['total_records_after'] = len(df_fixed)
        
        print(f"Processed {fixed_freq_stats['sequences_processed']} sequences")
        print(f"Time adjustments made: {fixed_freq_stats['time_adjustments']}")
        print(f"Glucose interpolations: {fixed_freq_stats['glucose_interpolations']}")
        print(f"Insulin records shifted: {fixed_freq_stats['insulin_shifted_records']}")
        print(f"Carb records shifted: {fixed_freq_stats['carb_shifted_records']}")
        print(f"Records before: {fixed_freq_stats['total_records_before']:,}")
        print(f"Records after: {fixed_freq_stats['total_records_after']:,}")
        print("Fixed-frequency data creation complete")
        
        return df_fixed, fixed_freq_stats

    def filter_glucose_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filter to keep only glucose data with simplified fields.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Tuple of (filtered DataFrame with only glucose data, filtering statistics)
        """
        print("Filtering to glucose-only data...")
        
        filtering_stats = {
            'original_records': len(df),
            'glucose_only_enabled': self.glucose_only,
            'records_after_filtering': 0,
            'records_removed': 0,
            'fields_removed': []
        }
        
        if not self.glucose_only:
            print("Glucose-only filtering is disabled - keeping all data")
            filtering_stats['records_after_filtering'] = len(df)
            return df, filtering_stats
        
        # Filter to keep only rows with non-null glucose values
        df_filtered = df.filter(pl.col('Glucose Value (mg/dL)').is_not_null())
        
        # Remove specified fields
        fields_to_remove = ['Event Type', 'Insulin Value (u)', 'Carb Value (grams)']
        existing_fields_to_remove = [field for field in fields_to_remove if field in df_filtered.columns]
        
        if existing_fields_to_remove:
            df_filtered = df_filtered.drop(existing_fields_to_remove)
            filtering_stats['fields_removed'] = existing_fields_to_remove
        
        # Update statistics
        filtering_stats['records_after_filtering'] = len(df_filtered)
        filtering_stats['records_removed'] = len(df) - len(df_filtered)
        
        print(f"Original records: {filtering_stats['original_records']:,}")
        print(f"Records with glucose values: {filtering_stats['records_after_filtering']:,}")
        print(f"Records removed (no glucose): {filtering_stats['records_removed']:,}")
        if filtering_stats['fields_removed']:
            print(f"Fields removed: {', '.join(filtering_stats['fields_removed'])}")
        print("OK: Glucose-only filtering complete")
        
        return df_filtered, filtering_stats
    
    def prepare_ml_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare final DataFrame for machine learning with sequence_id as first column.
        
        Args:
            df: Processed DataFrame with sequence IDs
            
        Returns:
            Final DataFrame ready for ML training
        """
        print("Preparing final ML dataset...")
        
        # Convert timestamp back to string format for output and convert all numeric fields to Float64
        columns_to_cast = []
        
        # Always convert timestamp
        columns_to_cast.append(
            pl.col('timestamp').dt.strftime('%Y-%m-%dT%H:%M:%S').alias('Timestamp (YYYY-MM-DDThh:mm:ss)')
        )
        
        # Convert numeric fields to Float64 if they exist
        if 'Glucose Value (mg/dL)' in df.columns:
            columns_to_cast.append(pl.col('Glucose Value (mg/dL)').cast(pl.Float64, strict=False))
        if 'Insulin Value (u)' in df.columns:
            columns_to_cast.append(pl.col('Insulin Value (u)').cast(pl.Float64, strict=False))
        if 'Carb Value (grams)' in df.columns:
            columns_to_cast.append(pl.col('Carb Value (grams)').cast(pl.Float64, strict=False))
        
        df = df.with_columns(columns_to_cast)
        
        # Reorder columns with sequence_id first
        ml_columns = ['sequence_id'] + [col for col in df.columns if col not in ['sequence_id', 'timestamp']]
        ml_df = df.select(ml_columns)
        
        return ml_df
    
    def get_statistics(self, df: pl.DataFrame, gap_stats: Dict, interp_stats: Dict, removal_stats: Dict = None, filter_stats: Dict = None, replacement_stats: Dict = None, glucose_filter_stats: Dict = None, fixed_freq_stats: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the processed data.
        
        Args:
            df: Final processed DataFrame
            gap_stats: Gap detection statistics
            interp_stats: Interpolation statistics
            removal_stats: Calibration removal statistics
            filter_stats: Sequence filtering statistics
            replacement_stats: High/Low value replacement statistics
            glucose_filter_stats: Glucose-only filtering statistics
            fixed_freq_stats: Fixed-frequency data creation statistics
            
        Returns:
            Dictionary with comprehensive statistics
        """
        # Get date range from timestamp column if available
        date_range = {'start': 'N/A', 'end': 'N/A'}
        if 'Timestamp (YYYY-MM-DDThh:mm:ss)' in df.columns:
            valid_timestamps = df.filter(pl.col('Timestamp (YYYY-MM-DDThh:mm:ss)').is_not_null())
            if len(valid_timestamps) > 0:
                timestamps = valid_timestamps['Timestamp (YYYY-MM-DDThh:mm:ss)'].sort()
                date_range = {
                    'start': timestamps[0],
                    'end': timestamps[-1]
                }
        
        # Calculate sequence statistics
        sequence_counts = df.group_by('sequence_id').count().sort('sequence_id')
        sequence_lengths = sequence_counts['count'].to_list()
        
        stats = {
            'dataset_overview': {
                'total_records': len(df),
                'total_sequences': df['sequence_id'].n_unique(),
                'date_range': date_range,
                'original_records': getattr(self, '_original_record_count', len(df))
            },
            'sequence_analysis': {
                'sequence_lengths': {
                    'count': len(sequence_lengths),
                    'mean': sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
                    'std': np.std(sequence_lengths) if sequence_lengths else 0,
                    'min': min(sequence_lengths) if sequence_lengths else 0,
                    '25%': np.percentile(sequence_lengths, 25) if sequence_lengths else 0,
                    '50%': np.percentile(sequence_lengths, 50) if sequence_lengths else 0,
                    '75%': np.percentile(sequence_lengths, 75) if sequence_lengths else 0,
                    'max': max(sequence_lengths) if sequence_lengths else 0
                },
                'longest_sequence': max(sequence_lengths) if sequence_lengths else 0,
                'shortest_sequence': min(sequence_lengths) if sequence_lengths else 0,
                'sequences_by_length': dict(zip(*np.unique(sequence_lengths, return_counts=True)))
            },
            'gap_analysis': gap_stats,
            'interpolation_analysis': interp_stats,
            'calibration_removal_analysis': removal_stats if removal_stats else {},
            'filtering_analysis': filter_stats if filter_stats else {},
            'replacement_analysis': replacement_stats if replacement_stats else {},
            'fixed_frequency_analysis': fixed_freq_stats if fixed_freq_stats else {},
            'glucose_filtering_analysis': glucose_filter_stats if glucose_filter_stats else {},
            'data_quality': {
                'glucose_data_completeness': (1 - df['Glucose Value (mg/dL)'].null_count() / len(df)) * 100 if 'Glucose Value (mg/dL)' in df.columns else 0,
                'insulin_data_completeness': (1 - df['Insulin Value (u)'].null_count() / len(df)) * 100 if 'Insulin Value (u)' in df.columns else 0,
                'carb_data_completeness': (1 - df['Carb Value (grams)'].null_count() / len(df)) * 100 if 'Carb Value (grams)' in df.columns else 0,
                'interpolated_records': df.filter(pl.col('Event Type') == 'Interpolated').height if 'Event Type' in df.columns else 0
            }
        }
        
        return stats
    
    def process(self, csv_folder: str, output_file: str = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline with mandatory consolidation.
        
        Args:
            csv_folder: Path to folder containing CSV files (consolidation is mandatory)
            output_file: Optional path to save processed data
            
        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
        """
        print("Starting glucose data preprocessing for ML...")
        print(f"Time discretization interval: {self.expected_interval_minutes} minutes")
        print(f"Small gap max (interpolation limit): {self.small_gap_max_minutes} minutes")
        print(f"Save intermediate files: {self.save_intermediate_files}")
        print("-" * 50)
        
        # Step 1: Consolidate CSV files (mandatory step)
        if self.save_intermediate_files:
            consolidated_file = "consolidated_data.csv"
        else:
            consolidated_file = None
        
        print("STEP 1: Consolidating CSV files (mandatory step)...")
        df = self.consolidate_glucose_data(csv_folder, consolidated_file)
        
        if self.save_intermediate_files:
            print(f"Consolidated data saved to: {consolidated_file}")
        
        print("-" * 40)
        
        # Step 2: Detect gaps and create sequences
        print("STEP 2: Detecting gaps and creating sequences...")
        df, gap_stats = self.detect_gaps_and_sequences(df)
        print("OK: Gap detection and sequence creation complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step2_sequences_created.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"Data with sequences saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 3: Interpolate missing values
        print("STEP 3: Interpolating missing values...")
        df, interp_stats = self.interpolate_missing_values(df)
        print("OK: Missing value interpolation complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step3_interpolated_values.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"Data with interpolated values saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 4: Filter sequences by minimum length
        print("STEP 4: Filtering sequences by minimum length...")
        df, filter_stats = self.filter_sequences_by_length(df)
        print("OK: Sequence filtering complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step4_filtered_sequences.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"Filtered data saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 5: Create fixed-frequency data (if enabled)
        if self.create_fixed_frequency:
            print("STEP 5: Creating fixed-frequency data...")
            df, fixed_freq_stats = self.create_fixed_frequency_data(df)
            print("Fixed-frequency data creation complete")
            
            if self.save_intermediate_files:
                intermediate_file = "step5_fixed_frequency.csv"
                df.write_csv(intermediate_file, null_value="")
                print(f"Fixed-frequency data saved to: {intermediate_file}")
        else:
            print("STEP 5: Fixed-frequency data creation is disabled - skipping")
            fixed_freq_stats = {}
        
        print("-" * 40)
        
        # Step 6: Filter to glucose-only data (if requested)
        print("STEP 6: Filtering to glucose-only data...")
        df, glucose_filter_stats = self.filter_glucose_only(df)
        print("OK: Glucose-only filtering complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step6_glucose_only.csv"
            df.write_csv(intermediate_file, null_value="")
            print(f"Glucose-only data saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Step 7: Prepare final ML dataset
        print("STEP 7: Preparing final ML dataset...")
        ml_df = self.prepare_ml_data(df)
        print("OK: ML dataset preparation complete")
        
        if self.save_intermediate_files:
            intermediate_file = "step7_ml_ready.csv"
            ml_df.write_csv(intermediate_file, null_value="")
            print(f"ML-ready data saved to: {intermediate_file}")
        
        print("-" * 40)
        
        # Generate statistics (removed database-specific stats that are now handled by converters)
        stats = self.get_statistics(ml_df, gap_stats, interp_stats, None, filter_stats, None, glucose_filter_stats, fixed_freq_stats)
        
        # Save final output if specified
        if output_file:
            ml_df.write_csv(output_file, null_value="")
            print(f"Final processed data saved to: {output_file}")
        
        print("-" * 50)
        print("Preprocessing completed successfully!")
        
        return ml_df, stats
    


def print_statistics(stats: Dict[str, Any], preprocessor: 'GlucoseMLPreprocessor' = None) -> None:
    """
    Print formatted statistics about the processed data.
    
    Args:
        stats: Statistics dictionary from preprocessor
        preprocessor: Optional preprocessor instance to show parameters
    """
    print("\n" + "="*60)
    print("GLUCOSE DATA PREPROCESSING STATISTICS")
    print("="*60)
    
    # Show parameters if preprocessor is provided
    if preprocessor:
        print(f"\nPARAMETERS USED:")
        print(f"   Time Discretization Interval: {preprocessor.expected_interval_minutes} minutes")
        print(f"   Small Gap Max (Interpolation Limit): {preprocessor.small_gap_max_minutes} minutes")
        print(f"   Remove Calibration Events: {preprocessor.remove_calibration}")
        print(f"   Minimum Sequence Length: {preprocessor.min_sequence_len}")
        print(f"   Calibration Period Threshold: {preprocessor.calibration_period_minutes} minutes")
        print(f"   Remove After Calibration: {preprocessor.remove_after_calibration_hours} hours")
        print(f"   Create Fixed-Frequency Data: {preprocessor.create_fixed_frequency}")
    
    # Dataset Overview
    overview = stats['dataset_overview']
    print(f"\nDATASET OVERVIEW:")
    print(f"   Total Records: {overview['total_records']:,}")
    print(f"   Total Sequences: {overview['total_sequences']:,}")
    print(f"   Date Range: {overview['date_range']['start']} to {overview['date_range']['end']}")
    
    # Show data preservation percentage
    original_records = overview.get('original_records', overview['total_records'])
    final_records = overview['total_records']
    preservation_percentage = (final_records / original_records * 100) if original_records > 0 else 100
    print(f"   Data Preservation: {preservation_percentage:.1f}% ({final_records:,}/{original_records:,} records)")
    
    # Sequence Analysis
    seq_analysis = stats['sequence_analysis']
    print(f"\nSEQUENCE ANALYSIS:")
    print(f"   Longest Sequence: {seq_analysis['longest_sequence']:,} records")
    print(f"   Shortest Sequence: {seq_analysis['shortest_sequence']:,} records")
    print(f"   Average Sequence Length: {seq_analysis['sequence_lengths']['mean']:.1f} records")
    print(f"   Median Sequence Length: {seq_analysis['sequence_lengths']['50%']:.1f} records")
    
    # Gap Analysis
    gap_analysis = stats['gap_analysis']
    print(f"\nGAP ANALYSIS:")
    print(f"   Total Gaps > {preprocessor.small_gap_max_minutes if preprocessor else 'N/A'} minutes: {gap_analysis['total_gaps']:,}")
    print(f"   Sequences Created: {gap_analysis['total_sequences']:,}")
    
    # Calibration Period Analysis
    if 'calibration_period_analysis' in gap_analysis:
        calib_analysis = gap_analysis['calibration_period_analysis']
        print(f"\nCALIBRATION PERIOD ANALYSIS:")
        print(f"   Calibration Periods Detected: {calib_analysis['calibration_periods_detected']:,}")
        print(f"   Records Removed After Calibration: {calib_analysis['total_records_marked_for_removal']:,}")
        print(f"   Sequences Affected: {calib_analysis['sequences_marked_for_removal']:,}")
    
    # High/Low Value Replacement Analysis
    if 'replacement_analysis' in stats and stats['replacement_analysis']:
        replacement_analysis = stats['replacement_analysis']
        print(f"\nHIGH/LOW VALUE REPLACEMENT ANALYSIS:")
        print(f"   High Values Replaced (-> 401): {replacement_analysis['high_replacements']:,}")
        print(f"   Low Values Replaced (-> 39): {replacement_analysis['low_replacements']:,}")
        print(f"   Total Replacements: {replacement_analysis['total_replacements']:,}")
        print(f"   Glucose Field Type: {'Float64' if replacement_analysis['glucose_field_converted_to_float'] else 'String'}")
    
    # Interpolation Analysis
    interp_analysis = stats['interpolation_analysis']
    print(f"\nINTERPOLATION ANALYSIS:")
    print(f"   Small Gaps Identified and Processed: {interp_analysis['small_gaps_filled']:,}")
    print(f"   Interpolated Data Points Created: {interp_analysis['total_interpolated_data_points']:,}")
    print(f"   Total Field Interpolations: {interp_analysis['total_interpolations']:,}")
    print(f"   Glucose Interpolations: {interp_analysis['glucose_value_mg/dl_interpolations']:,}")
    print(f"   Insulin Interpolations: {interp_analysis['insulin_value_u_interpolations']:,}")
    print(f"   Carb Interpolations: {interp_analysis['carb_value_grams_interpolations']:,}")
    print(f"   Large Gaps Skipped: {interp_analysis['large_gaps_skipped']:,}")
    print(f"   Sequences Processed: {interp_analysis['sequences_processed']:,}")
    
    # Calibration Removal Analysis
    if 'calibration_removal_analysis' in stats and stats['calibration_removal_analysis']:
        removal_analysis = stats['calibration_removal_analysis']
        print(f"\nCALIBRATION REMOVAL ANALYSIS:")
        print(f"   Calibration Events Removed: {removal_analysis.get('calibration_events_removed', 0):,}")
        print(f"   Records Before Removal: {removal_analysis.get('records_before_removal', 0):,}")
        print(f"   Records After Removal: {removal_analysis.get('records_after_removal', 0):,}")
        print(f"   Removal Enabled: {removal_analysis.get('calibration_removal_enabled', False)}")
    
    # Filtering Analysis
    if 'filtering_analysis' in stats and stats['filtering_analysis']:
        filter_analysis = stats['filtering_analysis']
        print(f"\nSEQUENCE FILTERING ANALYSIS:")
        print(f"   Original Sequences: {filter_analysis.get('original_sequences', 0):,}")
        print(f"   Sequences After Filtering: {filter_analysis.get('filtered_sequences', 0):,}")
        print(f"   Sequences Removed: {filter_analysis.get('removed_sequences', 0):,}")
        print(f"   Original Records: {filter_analysis.get('original_records', 0):,}")
        print(f"   Records After Filtering: {filter_analysis.get('filtered_records', 0):,}")
        print(f"   Records Removed: {filter_analysis.get('removed_records', 0):,}")
    
    # Fixed-Frequency Analysis
    if 'fixed_frequency_analysis' in stats and stats['fixed_frequency_analysis']:
        fixed_freq_analysis = stats['fixed_frequency_analysis']
        print(f"\nFIXED-FREQUENCY ANALYSIS:")
        print(f"   Sequences Processed: {fixed_freq_analysis.get('sequences_processed', 0):,}")
        print(f"   Time Adjustments Made: {fixed_freq_analysis.get('time_adjustments', 0):,}")
        print(f"   Glucose Interpolations: {fixed_freq_analysis.get('glucose_interpolations', 0):,}")
        print(f"   Insulin Records Shifted: {fixed_freq_analysis.get('insulin_shifted_records', 0):,}")
        print(f"   Carb Records Shifted: {fixed_freq_analysis.get('carb_shifted_records', 0):,}")
        print(f"   Records Before: {fixed_freq_analysis.get('total_records_before', 0):,}")
        print(f"   Records After: {fixed_freq_analysis.get('total_records_after', 0):,}")
    
    # Glucose Filtering Analysis
    if 'glucose_filtering_analysis' in stats and stats['glucose_filtering_analysis']:
        glucose_filter_analysis = stats['glucose_filtering_analysis']
        print(f"\nGLUCOSE-ONLY FILTERING ANALYSIS:")
        print(f"   Glucose-Only Mode Enabled: {glucose_filter_analysis.get('glucose_only_enabled', False)}")
        print(f"   Original Records: {glucose_filter_analysis.get('original_records', 0):,}")
        print(f"   Records After Filtering: {glucose_filter_analysis.get('records_after_filtering', 0):,}")
        print(f"   Records Removed (No Glucose): {glucose_filter_analysis.get('records_removed', 0):,}")
        if glucose_filter_analysis.get('fields_removed', []):
            print(f"   Fields Removed: {', '.join(glucose_filter_analysis['fields_removed'])}")
    
    # Data Quality
    quality = stats['data_quality']
    print(f"\nDATA QUALITY:")
    print(f"   Glucose Data Completeness: {quality['glucose_data_completeness']:.1f}%")
    print(f"   Insulin Data Completeness: {quality['insulin_data_completeness']:.1f}%")
    print(f"   Carb Data Completeness: {quality['carb_data_completeness']:.1f}%")
    print(f"   Interpolated Records: {quality['interpolated_records']:,}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = "zendo_small"  # Folder containing data files (can be any supported database type)
    OUTPUT_FILE = "glucose_ml_ready_test.csv"  # Final ML-ready output
    CONFIG_FILE = "glucose_config_new.yaml"  # Configuration file
    
    # Initialize preprocessor from configuration file
    try:
        preprocessor = GlucoseMLPreprocessor.from_config_file(CONFIG_FILE)
        print(f"Loaded configuration from: {CONFIG_FILE}")
    except FileNotFoundError:
        print(f"Configuration file {CONFIG_FILE} not found, using default settings")
        # Fallback to default configuration
    preprocessor = GlucoseMLPreprocessor(
        expected_interval_minutes=5,   # Time discretization interval
        small_gap_max_minutes=15,      # Maximum gap size to interpolate
        remove_calibration=True,       # Remove calibration events to create interpolatable gaps
        min_sequence_len=200,          # Minimum sequence length to keep for ML training
        calibration_period_minutes=60*2 + 45,  # Gap duration considered as calibration period (2h 45m)
        remove_after_calibration_hours=24      # Hours of data to remove after calibration period
    )
    
    # Process data
    try:
        # Start from data folder and consolidate (database type auto-detected)
        print("Starting glucose data processing...")
        print(f"Data folder: {DATA_FOLDER}")
        print(f"Output file: {OUTPUT_FILE}")
        print("-" * 50)
       
        ml_data, statistics = preprocessor.process(DATA_FOLDER, OUTPUT_FILE)
        
        # Print statistics
        print_statistics(statistics, preprocessor)
        
        # Show sample of processed data
        print(f"\nSAMPLE OF PROCESSED DATA:")
        print(ml_data.head(10))
        
        print(f"\nOutput file: {OUTPUT_FILE}")
        print(f"Ready for machine learning training!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
