#!/usr/bin/env python3
"""
Database converters for different glucose monitoring database types.

This module provides converters that handle the consolidation and processing
of different database structures (mono-user vs multi-user).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
from .base_converter import CSVFormatConverter
from .format_detector import CSVFormatDetector


class DatabaseConverter(ABC):
    """Base class for database converters."""
    
    def __init__(self, config: Dict[str, Any], output_fields: Optional[List[str]] = None):
        """
        Initialize the database converter.
        
        Args:
            config: Configuration dictionary with database-specific settings
            output_fields: List of field names to include in converter output.
                          If None, uses default fields matching current usage.
        """
        self.config = config
        self.output_fields = output_fields
        self.format_detector = CSVFormatDetector(output_fields)
    
    @abstractmethod
    def consolidate_data(self, data_folder: str, output_file: Optional[str] = None) -> pl.DataFrame:
        """
        Consolidate data from the database folder.
        
        Args:
            data_folder: Path to folder containing data files
            output_file: Optional path to save consolidated data
            
        Returns:
            Consolidated DataFrame
        """
        pass
    
    def _enforce_output_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Enforce that all default output fields are present in the DataFrame.
        Adds missing columns with null/empty string values to ensure schema consistency.
        
        Args:
            df: DataFrame to enforce schema on
            
        Returns:
            DataFrame with all default output fields present
        """
        # Get default output fields (display names) from CSVFormatConverter (config-based or default)
        default_fields = CSVFormatConverter.get_default_output_fields()
        
        # Add user_id for multi-user databases (it's added during processing)
        # Check if user_id column exists - if so, include it in required fields
        required_fields = default_fields.copy()
        if 'user_id' in df.columns:
            # user_id is already present, keep it
            pass
        
        # Add missing columns with appropriate null values
        for field in required_fields:
            if field not in df.columns:
                # Determine appropriate null value based on field type
                # String fields get empty string, numeric fields get null
                if 'Value' in field or 'grams' in field.lower():
                    # Numeric fields: use null (will be cast later)
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(field))
                else:
                    # String fields: use empty string
                    df = df.with_columns(pl.lit("").alias(field))
        
        # Ensure columns are in the correct order: timestamp first, then other fields
        # Keep any extra columns (like user_id) at the end
        existing_columns = df.columns
        ordered_columns = []
        
        # Add required fields in order
        for field in required_fields:
            if field in existing_columns:
                ordered_columns.append(field)
        
        # Add any remaining columns (like user_id, timestamp internal column, etc.)
        for col in existing_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        # Reorder columns
        df = df.select(ordered_columns)
        
        return df
    
    @abstractmethod
    def get_database_name(self) -> str:
        """
        Get the name of the database type this converter handles.
        
        Returns:
            String name of the database type
        """
        pass


class MonoUserDatabaseConverter(DatabaseConverter):
    """Converter for mono-user databases (Dexcom, Libre3)."""
    
    def consolidate_data(self, data_folder: str, output_file: Optional[str] = None) -> pl.DataFrame:
        """
        Consolidate mono-user data from multiple CSV files.
        
        Args:
            data_folder: Path to folder containing CSV files
            output_file: Optional path to save consolidated data
            
        Returns:
            Consolidated DataFrame with processed data
        """
        csv_path = Path(data_folder)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        if not csv_path.is_dir():
            raise ValueError(f"Input must be a directory containing CSV files, got: {data_folder}")
        
        all_data = []
        
        # Get all CSV files (including in subdirectories, sorted for deterministic processing order)
        csv_files = sorted(csv_path.glob("**/*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {data_folder}")
        
        print(f"Found {len(csv_files)} CSV files to consolidate")
        
        for csv_file in csv_files:
            print(f"Processing: {csv_file.name}")
            file_data = self._process_csv_file(csv_file)
            all_data.extend(file_data)
            print(f"  OK: Extracted {len(file_data)} records")
        
        print(f"\nTotal records collected: {len(all_data):,}")
        
        if not all_data:
            raise ValueError("No valid data found in CSV files!")
        
        # Ensure all records have all required fields before DataFrame creation
        # This prevents Polars from dropping columns when some records don't have them
        # Use empty strings for all fields to ensure consistent string type
        default_fields = CSVFormatConverter.get_default_output_fields()
        for record in all_data:
            for field in default_fields:
                if field not in record:
                    # Add missing field with empty string (consistent string type)
                    record[field] = ""
        
        # Convert to DataFrame for easier sorting
        df = pl.DataFrame(all_data)
        
        # Enforce output schema to ensure all default fields are present
        df = self._enforce_output_schema(df)
        
        # Parse timestamps and sort
        print("Parsing timestamps and sorting...")
        df = df.with_columns(
            pl.col('Timestamp (YYYY-MM-DDThh:mm:ss)').map_elements(self._parse_timestamp, return_dtype=pl.Datetime).alias('parsed_timestamp')
        )
        
        # Remove rows where timestamp parsing failed
        df = df.filter(pl.col('parsed_timestamp').is_not_null())
        
        print(f"Records with valid timestamps: {len(df):,}")
        
        # Sort by timestamp (oldest first)
        df = df.sort('parsed_timestamp')
        
        # Rename parsed_timestamp to timestamp for consistency
        df = df.rename({'parsed_timestamp': 'timestamp'})
        
        # Apply database-specific processing
        df = self._apply_database_specific_processing(df)
        
        # Write to output file
        if output_file:
            print(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        
        print(f"OK: Consolidation complete!")
        print(f"Total records in output: {len(df):,}")
        
        # Show date range
        if len(df) > 0:
            first_date = df['Timestamp (YYYY-MM-DDThh:mm:ss)'][0]
            last_date = df['Timestamp (YYYY-MM-DDThh:mm:ss)'][-1]
            print(f"Date range: {first_date} to {last_date}")

        return df
    
    def _process_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single CSV file and extract required fields using format detection."""
        data = []
        
        try:
            # Detect the format of the CSV file
            converter = self.format_detector.detect_format(file_path)
            
            if converter is None:
                print(f"Warning: Could not detect format for {file_path}, skipping file")
                return data
            
            print(f"Detected format: {converter.get_format_name()} for {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
                lines = file.readlines()
                
                # Find the line with headers
                header_line_num = None
                for line_num in range(min(3, len(lines))):
                    line = lines[line_num].strip()
                    if not line:
                        continue
                    
                    # Parse CSV line properly to handle quoted headers
                    import csv
                    from io import StringIO
                    csv_reader = csv.reader(StringIO(line))
                    headers = next(csv_reader)
                    
                    # Clean headers: remove quotes and strip whitespace
                    headers = [col.strip().strip('"') for col in headers]
                    
                    if converter.can_handle(headers):
                        header_line_num = line_num
                        break
                
                if header_line_num is None:
                    print(f"Could not find headers for {file_path}")
                    return data
                
                # Create a new file-like object starting from the header line
                from io import StringIO
                csv_content = ''.join(lines[header_line_num:])
                csv_file = StringIO(csv_content)
                
                import csv
                reader = csv.DictReader(csv_file)
                
                for row in reader:
                    # Use the appropriate converter to process the row
                    converted_record = converter.convert_row(row)
                    if converted_record is not None:
                        data.append(converted_record)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return data
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
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
    
    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply database-specific processing (to be overridden by subclasses).
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        return df
    
    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Mono-User Database"


class MultiUserDatabaseConverter(DatabaseConverter):
    """Converter for multi-user databases (UoM, Zendo)."""
    
    def consolidate_data(self, data_folder: str, output_file: Optional[str] = None) -> pl.DataFrame:
        """
        Consolidate multi-user data from multiple CSV files.
        
        Args:
            data_folder: Path to folder containing CSV files organized by user/data type
            output_file: Optional path to save consolidated data
            
        Returns:
            Consolidated DataFrame with user ID tracking
        """
        data_path = Path(data_folder)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        if not data_path.is_dir():
            raise ValueError(f"Input must be a directory, got: {data_folder}")
        
        all_user_data = []
        
        # Process each user separately (sorted for deterministic processing order)
        users_processed = self._identify_users(data_path)
        print(f"Found {len(users_processed)} users to process")
        
        for user_id, user_files in sorted(users_processed.items()):
            print(f"\nProcessing user: {user_id}")
            user_data = self._process_user_data(user_id, user_files)
            if user_data:
                all_user_data.extend(user_data)
                print(f"  OK: Extracted {len(user_data)} records for user {user_id}")
        
        print(f"\nTotal records collected: {len(all_user_data):,}")
        
        if not all_user_data:
            raise ValueError("No valid data found in data files!")
        
        # Ensure all records have all required fields before DataFrame creation
        # This prevents Polars from dropping columns when some records don't have them
        # Use empty strings for all fields to ensure consistent string type
        default_fields = CSVFormatConverter.get_default_output_fields()
        for record in all_user_data:
            for field in default_fields:
                if field not in record:
                    # Add missing field with empty string (consistent string type)
                    record[field] = ""
        
        # Convert to DataFrame
        df = pl.DataFrame(all_user_data)
        
        # Enforce output schema to ensure all default fields are present (should already be there, but double-check)
        df = self._enforce_output_schema(df)
        
        # Parse timestamps and sort by user and timestamp
        print("Parsing timestamps and sorting by user and time...")
        df = df.with_columns(
            pl.col('Timestamp (YYYY-MM-DDThh:mm:ss)').map_elements(self._parse_timestamp, return_dtype=pl.Datetime).alias('parsed_timestamp')
        )
        
        # Remove rows where timestamp parsing failed
        df = df.filter(pl.col('parsed_timestamp').is_not_null())
        
        print(f"Records with valid timestamps: {len(df):,}")
        
        # Sort by user_id and timestamp (each user's data sorted individually)
        df = df.sort(['user_id', 'parsed_timestamp'])
        
        # Rename parsed_timestamp to timestamp for consistency
        df = df.rename({'parsed_timestamp': 'timestamp'})
        
        # Write to output file
        if output_file:
            print(f"Writing consolidated data to: {output_file}")
            df.write_csv(output_file)
        
        print(f"OK: Multi-user consolidation complete!")
        print(f"Total records in output: {len(df):,}")
        
        # Show user statistics
        user_counts = df.group_by('user_id').count().sort('user_id')
        print(f"Users processed: {len(user_counts)}")
        for row in user_counts.iter_rows(named=True):
            print(f"  User {row['user_id']}: {row['count']:,} records")

        return df
    
    def _identify_users(self, data_path: Path) -> Dict[str, List[Path]]:
        """
        Identify users and their associated files.
        
        Args:
            data_path: Path to the data folder
            
        Returns:
            Dictionary mapping user_id to list of file paths
        """
        users = {}
        
        # Get all CSV files (sorted for deterministic processing order)
        csv_files = sorted(data_path.glob("**/*.csv"))
        
        for csv_file in csv_files:
            user_id = self._extract_user_id_from_filename(csv_file)
            if user_id:
                if user_id not in users:
                    users[user_id] = []
                users[user_id].append(csv_file)
        
        # Sort files within each user for deterministic processing
        for user_id in users:
            users[user_id] = sorted(users[user_id])
        
        return users
    
    def _extract_user_id_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Extract user ID from filename (to be overridden by subclasses).
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            User ID string or None if not found
        """
        return None
    
    def _process_user_data(self, user_id: str, user_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Process all files for a single user.
        
        Args:
            user_id: User identifier
            user_files: List of file paths for this user
            
        Returns:
            List of processed records for this user
        """
        user_data = []
        
        # Sort files for deterministic processing order
        for file_path in sorted(user_files):
            print(f"  Processing: {file_path.name}")
            file_data = self._process_csv_file(file_path, user_id)
            user_data.extend(file_data)
        
        return user_data
    
    def _process_csv_file(self, file_path: Path, user_id: str) -> List[Dict[str, Any]]:
        """Process a single CSV file and extract required fields using format detection."""
        data = []
        
        try:
            # Detect the format of the CSV file
            converter = self.format_detector.detect_format(file_path)
            
            if converter is None:
                print(f"Warning: Could not detect format for {file_path}, skipping file")
                return data
            
            print(f"    Detected format: {converter.get_format_name()} for {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
                lines = file.readlines()
                
                # Find the line with headers
                header_line_num = None
                for line_num in range(min(3, len(lines))):
                    line = lines[line_num].strip()
                    if not line:
                        continue
                    
                    # Parse CSV line properly to handle quoted headers
                    import csv
                    from io import StringIO
                    csv_reader = csv.reader(StringIO(line))
                    headers = next(csv_reader)
                    
                    # Clean headers: remove quotes and strip whitespace
                    headers = [col.strip().strip('"') for col in headers]
                    
                    if converter.can_handle(headers):
                        header_line_num = line_num
                        break
                
                if header_line_num is None:
                    print(f"Could not find headers for {file_path}")
                    return data
                
                # Create a new file-like object starting from the header line
                from io import StringIO
                csv_content = ''.join(lines[header_line_num:])
                csv_file = StringIO(csv_content)
                
                import csv
                reader = csv.DictReader(csv_file)
                
                for row in reader:
                    # Use the appropriate converter to process the row
                    converted_record = converter.convert_row(row)
                    if converted_record is not None:
                        # Add user_id to the record
                        converted_record['user_id'] = user_id
                        data.append(converted_record)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return data
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object for sorting."""
        if not timestamp_str or timestamp_str.strip() == "":
            return None
        
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
    
    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Multi-User Database"
