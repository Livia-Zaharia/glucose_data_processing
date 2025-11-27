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
        
        # Initialize statistics for calibration period analysis
        calibration_stats = {
            'calibration_periods_detected': 0,
            'sequences_marked_for_removal': 0,  # Not used in current logic but kept for structure
            'total_records_marked_for_removal': 0
        }
        
        # Handle multi-user data by processing each user separately
        if 'user_id' in df.columns:
            print("Processing multi-user data - creating sequences per user...")
            all_sequences = []
            
            for user_id in df['user_id'].unique():
                user_data = df.filter(pl.col('user_id') == user_id).sort('timestamp')
                user_sequences, user_calib_stats = self._create_sequences_for_user(user_data, user_id)
                all_sequences.append(user_sequences)
                
                # Aggregate stats
                calibration_stats['calibration_periods_detected'] += user_calib_stats['calibration_periods_detected']
                calibration_stats['total_records_marked_for_removal'] += user_calib_stats['total_records_marked_for_removal']
            
            # Combine all user sequences
            if all_sequences:
                df = pl.concat(all_sequences).sort(['user_id', 'sequence_id', 'timestamp'])
            else:
                df = df.clear()
        else:
            # Single user data - process normally
            df = df.sort('timestamp')
            df, calibration_stats = self._create_sequences_for_user(df)
        
        # Calculate statistics
        # Use len() instead of count() for Polars 1.0+ compatibility
        if 'user_id' in df.columns:
            sequence_counts = df.group_by(['user_id', 'sequence_id']).len().sort(['user_id', 'sequence_id'])
        else:
            sequence_counts = df.group_by(['sequence_id']).len().sort('sequence_id')
            
        stats = {
            'total_sequences': df['sequence_id'].n_unique(),
            'gap_positions': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
            'total_gaps': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
            'sequence_lengths': dict(zip(sequence_counts['sequence_id'].to_list(), sequence_counts['len'].to_list())),
            'calibration_period_analysis': calibration_stats
        }
        
        print(f"Created {stats['total_sequences']} sequences")
        print(f"Found {stats['total_gaps']} gaps > {self.small_gap_max_minutes} minutes")
        
        if calibration_stats['calibration_periods_detected'] > 0:
            print(f"Detected {calibration_stats['calibration_periods_detected']} calibration periods")
            print(f"Removed {calibration_stats['total_records_marked_for_removal']} records after calibration")
        
        # Remove temporary columns
        columns_to_remove = ['time_diff_seconds', 'is_gap', 'is_calibration_gap', 'remove_due_to_calibration']
        df = df.drop([col for col in columns_to_remove if col in df.columns])
        
        return df, stats
    
    def _create_sequences_for_user(self, user_df: pl.DataFrame, user_id: str = None) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Create sequences for a single user's data and handle calibration periods.
        
        Args:
            user_df: DataFrame with data for a single user
            user_id: User ID (for multi-user data)
            
        Returns:
            Tuple of (DataFrame with sequence IDs added and calibration data removed, calibration statistics)
        """
        stats = {
            'calibration_periods_detected': 0,
            'total_records_marked_for_removal': 0
        }
        
        if len(user_df) == 0:
            return user_df, stats

        # Calculate time differences
        df = user_df.with_columns([
            pl.col('timestamp').diff().dt.total_seconds().alias('time_diff_seconds')
        ])
        
        # Identify calibration gaps
        # A gap is a calibration gap if it exceeds calibration_period_minutes
        df = df.with_columns([
            (pl.col('time_diff_seconds') > self.calibration_period_seconds).fill_null(False).alias('is_calibration_gap')
        ])
        
        # Identify standard gaps (for sequence breaking)
        # Any gap > small_gap_max_minutes breaks a sequence
        df = df.with_columns([
            (pl.col('time_diff_seconds') > self.small_gap_max_seconds).fill_null(False).alias('is_gap')
        ])
        
        stats['calibration_periods_detected'] = df['is_calibration_gap'].sum()
        
        # If calibration gaps exist, mark data for removal
        if stats['calibration_periods_detected'] > 0:
            # Get indices of calibration gaps
            calibration_indices = df.with_row_index().filter(pl.col('is_calibration_gap'))['index'].to_list()
            
            # Create a mask for removal
            # We can't easily do this fully vectorially in Polars without a complex window function or join
            # So we'll use a list of timestamps to remove or a boolean mask
            
            # Efficient approach: Create a list of removal windows
            removal_windows = []
            timestamps = df['timestamp'].to_list()
            
            for idx in calibration_indices:
                # The gap is BEFORE the row at idx. 
                # So the calibration ended roughly at timestamps[idx] (start of new data segment)
                # But technically the gap duration is time_diff_seconds.
                # We want to remove data starting from timestamps[idx] for X hours.
                
                start_removal = timestamps[idx]
                end_removal = start_removal + timedelta(hours=self.remove_after_calibration_hours)
                removal_windows.append((start_removal, end_removal))
            
            # Apply removal windows
            # We construct a boolean expression for keeping data
            # Keep if NOT in any removal window
            
            # To do this efficiently in Polars without iteration in the filter:
            # We can create a "remove" column initialized to False
            # But looping over windows is necessary if we have them.
            
            # However, for large datasets, looping might be slow. 
            # A clearer way:
            # 1. Extract rows that start a calibration period
            # 2. Create a validity mask
            
            # Let's try a slightly different approach using joins if possible, or just filter
            # Since number of calibration gaps is likely small, loop is acceptable for constructing filter
            
            # We'll create a 'remove_due_to_calibration' column
            # This part uses Python loop but over gaps, not all rows
            
            # Optimization: If too many gaps, this could be slow. But calibration gaps are rare (days/weeks).
            
            is_kept = np.ones(len(df), dtype=bool)
            ts_array = df['timestamp'].to_numpy()
            
            for start, end in removal_windows:
                # Find indices within this window
                # timestamps >= start and timestamps < end
                # start is inclusive because that's the first point after the gap
                mask = (ts_array >= start) & (ts_array < end)
                is_kept[mask] = False
                
            stats['total_records_marked_for_removal'] = (~is_kept).sum()
            
            # Apply filter
            df = df.with_columns(pl.lit(is_kept).alias('keep_record'))
            df = df.filter(pl.col('keep_record')).drop('keep_record')
            
        
        # Re-calculate gaps and sequences on filtered data
        # We need to recalculate time_diffs because removing rows might create new adjacent rows 
        # (though typically we remove blocks so the gap structure changes)
        # Actually, if we remove the 24h block after a gap, the "gap" effectively becomes larger 
        # or shifts. But we treat the remaining data as a new sequence start.
        
        if len(df) > 0:
            df = df.with_columns([
                pl.col('timestamp').diff().dt.total_seconds().alias('time_diff_seconds'),
            ]).with_columns([
                (pl.col('time_diff_seconds') > self.small_gap_max_seconds).fill_null(False).alias('is_gap'),
            ]).with_columns([
                pl.col('is_gap').cum_sum().alias('sequence_id')
            ])
        
        # For multi-user data, create unique sequence IDs by combining user_id and sequence_id
        if user_id is not None and len(df) > 0:
            # Create global sequence IDs by offsetting based on user
            # user_id * 100000 + sequence_id
            try:
                user_num = int(user_id)
                df = df.with_columns([
                    (pl.lit(user_num * 100000) + pl.col('sequence_id')).alias('sequence_id')
                ])
            except ValueError:
                # If user_id is not numeric, we can't easily offset. 
                # Just keep sequence_id as is (this might overlap if combined later without care)
                # But the main loop handles concatenation.
                pass
        
        return df, stats
    
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
            'fast_acting_insulin_value_u_interpolations': 0,
            'long_acting_insulin_value_u_interpolations': 0,
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
                                    'timestamp': interpolated_time,
                                    'sequence_id': seq_id
                                }
                                
                                # Initialize all other columns from the original data as None
                                for col in seq_pandas.columns:
                                    if col not in new_row:
                                        new_row[col] = None
                                        
                                # Set Event Type to Interpolated if it exists
                                if 'Event Type' in seq_pandas.columns:
                                    new_row['Event Type'] = 'Interpolated'
                                
                                # Add user_id if it exists in the original data
                                if 'user_id' in seq_pandas.columns:
                                    new_row['user_id'] = prev_row['user_id']
                                
                                # Linear interpolation for numeric columns
                                numeric_cols = ['Glucose Value (mg/dL)', 'Insulin Value (u)', 'Fast-Acting Insulin Value (u)', 'Long-Acting Insulin Value (u)', 'Carb Value (grams)']
                                interpolations_made = 0
                                for col in numeric_cols:
                                    # Skip columns that don't exist in the data
                                    if col not in seq_pandas.columns:
                                        continue
                                        
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
                                            
                                            # Update stats for specific column
                                            stat_key = f'{col.lower().replace(" ", "_").replace("(", "").replace(")", "")}_interpolations'
                                            # Handle special case for hyphens in fast-acting/long-acting
                                            if "fast-acting" in col.lower():
                                                stat_key = "fast_acting_insulin_value_u_interpolations"
                                            elif "long-acting" in col.lower():
                                                stat_key = "long_acting_insulin_value_u_interpolations"
                                                
                                            if stat_key in interpolation_stats:
                                                interpolation_stats[stat_key] += 1
                                                
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
        # Use len() instead of count() for Polars 1.0+ compatibility
        sequence_counts = df.group_by('sequence_id').len().sort('sequence_id')
        
        # Find sequences to keep (longer than or equal to min_sequence_len)
        sequences_to_keep = sequence_counts.filter(pl.col('len') >= self.min_sequence_len)
        
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
            removed_sequences = sequence_counts.filter(pl.col('len') < self.min_sequence_len)
            if len(removed_sequences) > 0:
                min_len_removed = removed_sequences['len'].min()
                max_len_removed = removed_sequences['len'].max()
                avg_len_removed = removed_sequences['len'].mean()
                print(f"Removed sequence lengths - Min: {min_len_removed}, Max: {max_len_removed}, Avg: {avg_len_removed:.1f}")
        
        return filtered_df, filtering_stats
    
    def create_fixed_frequency_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Create fixed-frequency data by aligning sequences to round minutes and ensuring consistent intervals.
        Glucose values are interpolated, while carbs and insulin are shifted to closest datapoints.
        
        Uses a declarative Polars-native approach for better performance and maintainability.
        
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
        
        # Process each sequence using Polars-native operations
        unique_sequences = df['sequence_id'].unique().to_list()
        all_fixed_sequences = []
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col('sequence_id') == seq_id).sort('timestamp')
            
            if len(seq_data) < 2:
                # Keep single-point sequences as-is
                all_fixed_sequences.append(seq_data)
                continue
                
            fixed_freq_stats['sequences_processed'] += 1
            
            # Create fixed-frequency timestamps using Polars operations
            fixed_seq_data = self._create_fixed_frequency_sequence(seq_data, seq_id, fixed_freq_stats)
            all_fixed_sequences.append(fixed_seq_data)
        
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
    
    def _create_fixed_frequency_sequence(self, seq_data: pl.DataFrame, seq_id: int, stats: Dict[str, Any]) -> pl.DataFrame:
        """
        Create fixed-frequency data for a single sequence using efficient Polars operations.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            stats: Statistics dictionary to update
            
        Returns:
            Fixed-frequency sequence as Polars DataFrame
        """
        # Get first and last timestamps
        first_timestamp = seq_data['timestamp'].min()
        last_timestamp = seq_data['timestamp'].max()
        
        # Calculate aligned start time
        first_second = first_timestamp.second
        if first_second >= 30:
            adjustment_seconds = 60 - first_second
        else:
            adjustment_seconds = -first_second
        
        aligned_start = first_timestamp + timedelta(seconds=adjustment_seconds)
        
        if adjustment_seconds != 0:
            stats['time_adjustments'] += 1
        
        # Generate fixed-frequency timestamps using efficient approach
        total_duration = (last_timestamp - aligned_start).total_seconds()
        num_intervals = int(total_duration / (self.expected_interval_minutes * 60)) + 1
        
        # Create fixed timestamps using efficient list comprehension
        fixed_timestamps_list = [
            aligned_start + timedelta(minutes=i * self.expected_interval_minutes)
            for i in range(num_intervals)
            if aligned_start + timedelta(minutes=i * self.expected_interval_minutes) <= last_timestamp
        ]
        
        fixed_timestamps = pl.DataFrame({
            'timestamp': fixed_timestamps_list
        }).with_columns([
            pl.col('timestamp').dt.strftime('%Y-%m-%dT%H:%M:%S').alias('Timestamp (YYYY-MM-DDThh:mm:ss)'),
            pl.lit(seq_id).alias('sequence_id')
        ])
        
        # 1. Interpolate Glucose (Linear)
        result_df = self._interpolate_glucose_linear(fixed_timestamps, seq_data, stats)
        
        # 2. Shift Events (Nearest Neighbor / Rounding)
        # Include all potential event columns
        potential_event_cols = ['Insulin Value (u)', 'Fast-Acting Insulin Value (u)', 'Long-Acting Insulin Value (u)', 'Carb Value (grams)', 'Event Type', 'user_id']
        event_cols = [col for col in potential_event_cols if col in seq_data.columns]
        
        if event_cols:
            events_df = self._shift_events_rounding(seq_data, event_cols, stats)
            # Join events with result (left join to keep all fixed timestamps)
            result_df = result_df.join(events_df, on='timestamp', how='left')
        
        return result_df
    
    def _interpolate_glucose_linear(self, fixed_timestamps: pl.DataFrame, seq_data: pl.DataFrame, stats: Dict[str, Any]) -> pl.DataFrame:
        """
        Interpolate glucose values linearly using previous and next data points.
        """
        if 'Glucose Value (mg/dL)' not in seq_data.columns:
            return fixed_timestamps
            
        # Prepare sequence data with original timestamp preserved
        # Ensure glucose is float for interpolation
        seq_data_ts = seq_data.select(['timestamp', 'Glucose Value (mg/dL)']).with_columns([
            pl.col('timestamp').alias('ts_orig'),
            pl.col('Glucose Value (mg/dL)').cast(pl.Float64, strict=False)
        ]).filter(pl.col('Glucose Value (mg/dL)').is_not_null())
        
        if len(seq_data_ts) == 0:
            return fixed_timestamps.with_columns(pl.lit(None).alias('Glucose Value (mg/dL)'))

        # Forward join (finds next point)
        forward = fixed_timestamps.join_asof(
            seq_data_ts, 
            on='timestamp', 
            strategy='forward'
        ).rename({'Glucose Value (mg/dL)': 'glucose_next', 'ts_orig': 'ts_next'})
        
        # Backward join (finds prev point)
        backward = fixed_timestamps.join_asof(
            seq_data_ts, 
            on='timestamp', 
            strategy='backward'
        ).select(['timestamp', 'Glucose Value (mg/dL)', 'ts_orig']).rename({'Glucose Value (mg/dL)': 'glucose_prev', 'ts_orig': 'ts_prev'})
        
        # Combine
        combined = forward.join(backward, on='timestamp', how='left')
        
        # Calculate interpolation
        # y = y_prev + (y_next - y_prev) * (t - t_prev) / (t_next - t_prev)
        
        result = combined.with_columns([
            pl.when(pl.col('ts_prev') == pl.col('ts_next')) # Exact match (or only one point)
            .then(pl.col('glucose_prev'))
            .when(pl.col('ts_prev').is_null()) # No prev (start of series)
            .then(pl.col('glucose_next'))
            .when(pl.col('ts_next').is_null()) # No next (end of series)
            .then(pl.col('glucose_prev'))
            .otherwise(
                pl.col('glucose_prev') + (
                    (pl.col('glucose_next') - pl.col('glucose_prev')) * 
                    (pl.col('timestamp') - pl.col('ts_prev')).dt.total_seconds() / 
                    (pl.col('ts_next') - pl.col('ts_prev')).dt.total_seconds()
                )
            ).alias('Glucose Value (mg/dL)')
        ])
        
        # Update stats
        interpolated_count = result.filter(
            pl.col('Glucose Value (mg/dL)').is_not_null() & 
            (pl.col('ts_next') != pl.col('timestamp')) # Not an exact match
        ).height
        stats['glucose_interpolations'] += interpolated_count
        
        return result.select(fixed_timestamps.columns + ['Glucose Value (mg/dL)'])

    def _shift_events_rounding(self, seq_data: pl.DataFrame, cols: List[str], stats: Dict[str, Any]) -> pl.DataFrame:
        """
        Shift events to nearest grid point by rounding timestamps.
        Aggregates multiple events falling into the same bin (sum for numeric, first for string).
        """
        # Select relevant columns
        events = seq_data.select(['timestamp'] + cols)
        
        # Filter out rows where all event columns are null
        # This prevents "empty" rows from creating 0-valued event records
        events = events.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols]))
        
        # Round timestamp to nearest interval
        interval_str = f"{self.expected_interval_minutes}m"
        
        events_rounded = events.with_columns(
            pl.col('timestamp').dt.round(interval_str).alias('timestamp')
        )
        
        # Define aggregations
        agg_exprs = []
        for col in cols:
            if col in ['Carb Value (grams)', 'Insulin Value (u)', 'Fast-Acting Insulin Value (u)', 'Long-Acting Insulin Value (u)']:
                # Sum numeric events (e.g. two small boluses)
                agg_exprs.append(pl.col(col).sum().alias(col))
                # Count shifts? Hard to track individual shifts here efficiently without loop
                if col == 'Carb Value (grams)':
                    stats['carb_shifted_records'] += 1 
                elif 'Insulin' in col:
                    stats['insulin_shifted_records'] += 1
            else:
                # For categorical/IDs, take the first (or max?)
                agg_exprs.append(pl.col(col).first().alias(col))
        
        # Group and aggregate
        shifted = events_rounded.group_by('timestamp').agg(agg_exprs)
        
        return shifted

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
        fields_to_remove = ['Event Type', 'Insulin Value (u)', 'Fast-Acting Insulin Value (u)', 'Long-Acting Insulin Value (u)', 'Carb Value (grams)']
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
        if 'Fast-Acting Insulin Value (u)' in df.columns:
            columns_to_cast.append(pl.col('Fast-Acting Insulin Value (u)').cast(pl.Float64, strict=False))
        if 'Long-Acting Insulin Value (u)' in df.columns:
            columns_to_cast.append(pl.col('Long-Acting Insulin Value (u)').cast(pl.Float64, strict=False))
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
        # Use len() instead of count() for Polars 1.0+ compatibility
        sequence_counts = df.group_by('sequence_id').len().sort('sequence_id')
        sequence_lengths = sequence_counts['len'].to_list()
        
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
                'fast_acting_insulin_data_completeness': (1 - df['Fast-Acting Insulin Value (u)'].null_count() / len(df)) * 100 if 'Fast-Acting Insulin Value (u)' in df.columns else 0,
                'long_acting_insulin_data_completeness': (1 - df['Long-Acting Insulin Value (u)'].null_count() / len(df)) * 100 if 'Long-Acting Insulin Value (u)' in df.columns else 0,
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
    
    def process_multiple_databases(self, csv_folders: List[str], output_file: str = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Process multiple databases with different formats and combine them into a single output.
        Sequence IDs are tracked and offset to ensure consistency across databases.
        
        Note: The user_id column (present in multi-user databases like UoM) is automatically 
        removed to ensure schema compatibility when combining databases with different structures.
        
        Args:
            csv_folders: List of paths to folders containing CSV files (each can be different format)
            output_file: Optional path to save combined processed data
            
        Returns:
            Tuple of (combined DataFrame, aggregated statistics dictionary)
        """
        print(f"Starting multi-database processing for {len(csv_folders)} databases...")
        print(f"Databases to process: {', '.join(csv_folders)}")
        print("-" * 50)
        
        all_dataframes = []
        all_statistics = []
        cumulative_sequence_offset = 0
        
        for idx, csv_folder in enumerate(csv_folders, 1):
            print(f"\n{'=' * 60}")
            print(f"PROCESSING DATABASE {idx}/{len(csv_folders)}: {csv_folder}")
            print(f"{'=' * 60}\n")
            
            # Process this database (without saving final output yet)
            ml_df, stats = self.process(csv_folder, output_file=None)
            
            # Remove user_id column if present (to ensure consistent schema across databases)
            if 'user_id' in ml_df.columns:
                print(f"\n⚙️  Removing user_id column for multi-database compatibility...")
                ml_df = ml_df.drop('user_id')
            
            # Offset sequence IDs to ensure uniqueness across databases
            if cumulative_sequence_offset > 0:
                print(f"⚙️  Offsetting sequence IDs by {cumulative_sequence_offset} for database consistency...")
                ml_df = ml_df.with_columns([
                    (pl.col('sequence_id') + cumulative_sequence_offset).alias('sequence_id')
                ])
            
            # Track the maximum sequence ID for the next database
            max_sequence_id = ml_df['sequence_id'].max()
            cumulative_sequence_offset = max_sequence_id + 1
            
            # Add database identifier to statistics
            stats['database_info'] = {
                'database_index': idx,
                'database_path': csv_folder,
                'sequence_id_offset': cumulative_sequence_offset - (max_sequence_id + 1),
                'sequence_id_range': {
                    'min': ml_df['sequence_id'].min(),
                    'max': max_sequence_id
                }
            }
            
            all_dataframes.append(ml_df)
            all_statistics.append(stats)
            
            print(f"\n✅ Database {idx} processed: {len(ml_df):,} records, {ml_df['sequence_id'].n_unique():,} sequences")
            print(f"   Sequence ID range: {ml_df['sequence_id'].min()} - {max_sequence_id}")
        
        # Combine all DataFrames
        print(f"\n{'=' * 60}")
        print("COMBINING ALL DATABASES")
        print(f"{'=' * 60}\n")
        
        combined_df = pl.concat(all_dataframes)
        
        # Sort by sequence_id and timestamp (user_id is removed for multi-database consistency)
        combined_df = combined_df.sort(['sequence_id', 'Timestamp (YYYY-MM-DDThh:mm:ss)'])
        
        # Aggregate statistics from all databases
        combined_stats = self._aggregate_statistics(all_statistics, csv_folders)
        
        print(f"✅ Combined {len(csv_folders)} databases:")
        print(f"   Total records: {len(combined_df):,}")
        print(f"   Total sequences: {combined_df['sequence_id'].n_unique():,}")
        print(f"   Sequence ID range: {combined_df['sequence_id'].min()} - {combined_df['sequence_id'].max()}")
        
        # Save final output if specified
        if output_file:
            combined_df.write_csv(output_file, null_value="")
            print(f"\n💾 Final combined data saved to: {output_file}")
        
        print("-" * 50)
        print(f"Multi-database preprocessing completed successfully!")
        
        return combined_df, combined_stats
    
    def _aggregate_statistics(self, all_statistics: List[Dict[str, Any]], csv_folders: List[str]) -> Dict[str, Any]:
        """
        Aggregate statistics from multiple databases into a single comprehensive report.
        
        Args:
            all_statistics: List of statistics dictionaries from each database
            csv_folders: List of database folder paths
            
        Returns:
            Aggregated statistics dictionary
        """
        # Initialize aggregated statistics with multi-database info
        aggregated = {
            'multi_database_info': {
                'total_databases': len(all_statistics),
                'database_paths': csv_folders,
                'databases_processed': []
            },
            'dataset_overview': {
                'total_records': 0,
                'total_sequences': 0,
                'date_range': {'start': None, 'end': None},
                'original_records': 0
            },
            'sequence_analysis': {
                'sequence_lengths': {
                    'count': 0,
                    'mean': 0,
                    'std': 0,
                    'min': float('inf'),
                    '25%': 0,
                    '50%': 0,
                    '75%': 0,
                    'max': 0
                },
                'longest_sequence': 0,
                'shortest_sequence': float('inf'),
                'sequences_by_length': {}
            },
            'gap_analysis': {
                'total_sequences': 0,
                'gap_positions': 0,
                'total_gaps': 0,
                'sequence_lengths': {},
                'calibration_period_analysis': {
                    'calibration_periods_detected': 0,
                    'sequences_marked_for_removal': 0,
                    'total_records_marked_for_removal': 0
                }
            },
            'interpolation_analysis': {
                'total_interpolations': 0,
                'total_interpolated_data_points': 0,
                'glucose_value_mg/dl_interpolations': 0,
                'insulin_value_u_interpolations': 0,
                'carb_value_grams_interpolations': 0,
                'sequences_processed': 0,
                'small_gaps_filled': 0,
                'large_gaps_skipped': 0
            },
            'calibration_removal_analysis': {},
            'filtering_analysis': {
                'original_sequences': 0,
                'filtered_sequences': 0,
                'removed_sequences': 0,
                'original_records': 0,
                'filtered_records': 0,
                'removed_records': 0
            },
            'replacement_analysis': {},
            'fixed_frequency_analysis': {
                'sequences_processed': 0,
                'total_records_before': 0,
                'total_records_after': 0,
                'glucose_interpolations': 0,
                'carb_shifted_records': 0,
                'insulin_shifted_records': 0,
                'time_adjustments': 0
            },
            'glucose_filtering_analysis': {},
            'data_quality': {
                'glucose_data_completeness': 0,
                'insulin_data_completeness': 0,
                'carb_data_completeness': 0,
                'interpolated_records': 0
            }
        }
        
        # Aggregate statistics from each database
        all_sequence_lengths = []
        
        for idx, stats in enumerate(all_statistics):
            db_info = stats.get('database_info', {})
            db_info['database_name'] = csv_folders[idx]
            aggregated['multi_database_info']['databases_processed'].append(db_info)
            
            # Dataset overview
            overview = stats.get('dataset_overview', {})
            aggregated['dataset_overview']['total_records'] += overview.get('total_records', 0)
            aggregated['dataset_overview']['total_sequences'] += overview.get('total_sequences', 0)
            aggregated['dataset_overview']['original_records'] += overview.get('original_records', 0)
            
            # Update date range
            date_range = overview.get('date_range', {})
            if date_range.get('start'):
                if aggregated['dataset_overview']['date_range']['start'] is None:
                    aggregated['dataset_overview']['date_range']['start'] = date_range['start']
                else:
                    aggregated['dataset_overview']['date_range']['start'] = min(
                        aggregated['dataset_overview']['date_range']['start'],
                        date_range['start']
                    )
            
            if date_range.get('end'):
                if aggregated['dataset_overview']['date_range']['end'] is None:
                    aggregated['dataset_overview']['date_range']['end'] = date_range['end']
                else:
                    aggregated['dataset_overview']['date_range']['end'] = max(
                        aggregated['dataset_overview']['date_range']['end'],
                        date_range['end']
                    )
            
            # Sequence analysis
            seq_analysis = stats.get('sequence_analysis', {})
            seq_lengths = seq_analysis.get('sequence_lengths', {})
            
            # Collect sequence lengths for global statistics
            if 'sequence_lengths' in stats.get('gap_analysis', {}):
                sequence_lengths_dict = stats['gap_analysis']['sequence_lengths']
                all_sequence_lengths.extend(list(sequence_lengths_dict.values()))
            
            # Update min/max
            aggregated['sequence_analysis']['longest_sequence'] = max(
                aggregated['sequence_analysis']['longest_sequence'],
                seq_analysis.get('longest_sequence', 0)
            )
            
            if seq_analysis.get('shortest_sequence', float('inf')) < aggregated['sequence_analysis']['shortest_sequence']:
                aggregated['sequence_analysis']['shortest_sequence'] = seq_analysis.get('shortest_sequence', 0)
            
            # Gap analysis
            gap_analysis = stats.get('gap_analysis', {})
            aggregated['gap_analysis']['total_sequences'] += gap_analysis.get('total_sequences', 0)
            aggregated['gap_analysis']['total_gaps'] += gap_analysis.get('total_gaps', 0)
            
            # Calibration period analysis
            calib_analysis = gap_analysis.get('calibration_period_analysis', {})
            aggregated['gap_analysis']['calibration_period_analysis']['calibration_periods_detected'] += calib_analysis.get('calibration_periods_detected', 0)
            aggregated['gap_analysis']['calibration_period_analysis']['sequences_marked_for_removal'] += calib_analysis.get('sequences_marked_for_removal', 0)
            aggregated['gap_analysis']['calibration_period_analysis']['total_records_marked_for_removal'] += calib_analysis.get('total_records_marked_for_removal', 0)
            
            # Interpolation analysis
            interp_analysis = stats.get('interpolation_analysis', {})
            aggregated['interpolation_analysis']['total_interpolations'] += interp_analysis.get('total_interpolations', 0)
            aggregated['interpolation_analysis']['total_interpolated_data_points'] += interp_analysis.get('total_interpolated_data_points', 0)
            aggregated['interpolation_analysis']['glucose_value_mg/dl_interpolations'] += interp_analysis.get('glucose_value_mg/dl_interpolations', 0)
            aggregated['interpolation_analysis']['insulin_value_u_interpolations'] += interp_analysis.get('insulin_value_u_interpolations', 0)
            aggregated['interpolation_analysis']['carb_value_grams_interpolations'] += interp_analysis.get('carb_value_grams_interpolations', 0)
            aggregated['interpolation_analysis']['sequences_processed'] += interp_analysis.get('sequences_processed', 0)
            aggregated['interpolation_analysis']['small_gaps_filled'] += interp_analysis.get('small_gaps_filled', 0)
            aggregated['interpolation_analysis']['large_gaps_skipped'] += interp_analysis.get('large_gaps_skipped', 0)
            
            # Filtering analysis
            filter_analysis = stats.get('filtering_analysis', {})
            if filter_analysis:
                aggregated['filtering_analysis']['original_sequences'] += filter_analysis.get('original_sequences', 0)
                aggregated['filtering_analysis']['filtered_sequences'] += filter_analysis.get('filtered_sequences', 0)
                aggregated['filtering_analysis']['removed_sequences'] += filter_analysis.get('removed_sequences', 0)
                aggregated['filtering_analysis']['original_records'] += filter_analysis.get('original_records', 0)
                aggregated['filtering_analysis']['filtered_records'] += filter_analysis.get('filtered_records', 0)
                aggregated['filtering_analysis']['removed_records'] += filter_analysis.get('removed_records', 0)
            
            # Fixed-frequency analysis
            fixed_freq_analysis = stats.get('fixed_frequency_analysis', {})
            if fixed_freq_analysis:
                aggregated['fixed_frequency_analysis']['sequences_processed'] += fixed_freq_analysis.get('sequences_processed', 0)
                aggregated['fixed_frequency_analysis']['total_records_before'] += fixed_freq_analysis.get('total_records_before', 0)
                aggregated['fixed_frequency_analysis']['total_records_after'] += fixed_freq_analysis.get('total_records_after', 0)
                aggregated['fixed_frequency_analysis']['glucose_interpolations'] += fixed_freq_analysis.get('glucose_interpolations', 0)
                aggregated['fixed_frequency_analysis']['carb_shifted_records'] += fixed_freq_analysis.get('carb_shifted_records', 0)
                aggregated['fixed_frequency_analysis']['insulin_shifted_records'] += fixed_freq_analysis.get('insulin_shifted_records', 0)
                aggregated['fixed_frequency_analysis']['time_adjustments'] += fixed_freq_analysis.get('time_adjustments', 0)
        
        # Calculate aggregated sequence statistics
        if all_sequence_lengths:
            aggregated['sequence_analysis']['sequence_lengths']['count'] = len(all_sequence_lengths)
            aggregated['sequence_analysis']['sequence_lengths']['mean'] = np.mean(all_sequence_lengths)
            aggregated['sequence_analysis']['sequence_lengths']['std'] = np.std(all_sequence_lengths)
            aggregated['sequence_analysis']['sequence_lengths']['min'] = np.min(all_sequence_lengths)
            aggregated['sequence_analysis']['sequence_lengths']['25%'] = np.percentile(all_sequence_lengths, 25)
            aggregated['sequence_analysis']['sequence_lengths']['50%'] = np.percentile(all_sequence_lengths, 50)
            aggregated['sequence_analysis']['sequence_lengths']['75%'] = np.percentile(all_sequence_lengths, 75)
            aggregated['sequence_analysis']['sequence_lengths']['max'] = np.max(all_sequence_lengths)
        
        return aggregated


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
    
    # Show multi-database information if present
    if 'multi_database_info' in stats:
        multi_db_info = stats['multi_database_info']
        print(f"\nMULTI-DATABASE PROCESSING:")
        print(f"   Total Databases Combined: {multi_db_info['total_databases']}")
        print(f"   Database Paths:")
        for i, path in enumerate(multi_db_info['database_paths'], 1):
            print(f"      {i}. {path}")
        
        print(f"\n   Processed Databases Details:")
        for db in multi_db_info['databases_processed']:
            db_idx = db.get('database_index', 'N/A')
            db_name = db.get('database_name', 'Unknown')
            seq_range = db.get('sequence_id_range', {})
            print(f"      Database {db_idx} ({db_name}):")
            print(f"         Sequence ID Range: {seq_range.get('min', 'N/A')} - {seq_range.get('max', 'N/A')}")
    
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
