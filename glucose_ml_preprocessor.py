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
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import warnings
import yaml
import sys
import json

# Import database detection and conversion classes
from formats import DatabaseDetector
from formats.base_converter import CSVFormatConverter

warnings.filterwarnings('ignore')

# Constants for common field names and literal values
INTERPOLATED_EVENT_TYPE = 'Interpolated'
DEFAULT_STREAMING_MAX_BUFFER_MB = 256
DEFAULT_STREAMING_FLUSH_MAX_USERS = 10
MIN_BUFFER_MB = 32

class StandardFieldNames:
    """
    Standard field names for flexible field approach.
    Uses universal standard names (not display names) to support arbitrary fields.
    """
    
    # Core standard field names that the preprocessor knows about
    TIMESTAMP = 'timestamp'
    EVENT_TYPE = 'event_type'
    GLUCOSE_VALUE = 'glucose_value_mgdl'
    FAST_ACTING_INSULIN = 'fast_acting_insulin_u'
    LONG_ACTING_INSULIN = 'long_acting_insulin_u'
    CARB_VALUE = 'carb_grams'
    USER_ID = 'user_id'
    SEQUENCE_ID = 'sequence_id'
    
    def __init__(self) -> None:
        """Initialize standard field names."""
        # Get known standard fields from CSVFormatConverter
        self._known_fields = set(CSVFormatConverter.get_field_to_display_name_map().keys())
    
    def is_known_field(self, standard_name: str) -> bool:
        """
        Check if a standard field name is in the known fields set.
        
        Args:
            standard_name: Standard field name to check
            
        Returns:
            True if field is known
        """
        return standard_name in self._known_fields


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
    def from_config_file(cls, config_path: Path, **cli_overrides: Any) -> "GlucoseMLPreprocessor":
        """
        Create a GlucoseMLPreprocessor instance from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            **cli_overrides: Command line arguments that override config values
            
        Returns:
            GlucoseMLPreprocessor instance with loaded configuration
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Initialize CSVFormatConverter with field mappings from config
        CSVFormatConverter.initialize_from_config(config)
        
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
            config=config,
            first_n_users=cli_overrides.get('first_n_users', config.get('first_n_users', None))
        )
    
    def __init__(
        self,
        expected_interval_minutes: int = 5,
        small_gap_max_minutes: int = 15,
        remove_calibration: bool = True,
        min_sequence_len: int = 200,
        save_intermediate_files: bool = False,
        calibration_period_minutes: int = 60*2 + 45,
        remove_after_calibration_hours: int = 24,
        high_glucose_value: int = 401,
        low_glucose_value: int = 39,
        glucose_only: bool = False,
        create_fixed_frequency: bool = True,
        config: Optional[Dict[str, Any]] = None,
        first_n_users: Optional[int] = None,
    ) -> None:
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
        self.config = config if config is not None else {}
        if first_n_users is not None:
            self.config['first_n_users'] = first_n_users
        self.expected_interval_seconds = expected_interval_minutes * 60
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        self.calibration_period_seconds = calibration_period_minutes * 60
        self._original_record_count: int = 0
        self._field_categories_dict: Optional[Dict[str, Any]] = None
        
        # Initialize standard field names for universal field name handling
        self.fields = StandardFieldNames()
    
    @staticmethod
    def extract_field_categories(database_type: str) -> Dict[str, Any]:
        """
        Extract field categories and settings from schema file.
        
        Args:
            database_type: Database type (e.g., 'uom', 'dexcom', 'freestyle_libre3')
            
        Returns:
            Dictionary with categories ('continuous', 'occasional', 'service') 
            and settings (e.g., 'remove_after_calibration')
        """
        # Map database type to schema file name (legacy aliases).
        # Prefer convention: `<database_type>_schema.json` if present.
        schema_files = {
            'uom': 'uom_schema.json',
            'dexcom': 'dexcom_schema.json',
            'libre3': 'freestyle_libre3_schema.json',
            'freestyle_libre3': 'freestyle_libre3_schema.json',
        }

        schema_file = schema_files.get(database_type, f"{database_type}_schema.json")
        
        # Load schema file
        schema_path = Path(__file__).parent / 'formats' / schema_file
        if not schema_path.exists():
            # Return default with only glucose as continuous
            return {
                'continuous': [StandardFieldNames.GLUCOSE_VALUE],
                'occasional': [],
                'service': [],
                'remove_after_calibration': True
            }
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Get field_categories from schema
        field_categories = schema.get('field_categories', {})
        
        # Build result dictionary using standard field names directly
        result = {
            'continuous': [],
            'occasional': [],
            'service': [],
            'remove_after_calibration': schema.get('remove_after_calibration', True)
        }
        
        for standard_name, category in field_categories.items():
            if category in result:
                result[category].append(standard_name)
        
        # Always ensure glucose is in continuous (if it exists)
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        if glucose_col not in result['continuous']:
            result['continuous'].append(glucose_col)
        
        return result

    @staticmethod
    def _df_estimated_size_bytes(df: pl.DataFrame) -> int:
        """
        Best-effort DataFrame size estimate for buffering decisions.
        """
        try:
            return int(df.estimated_size())
        except (AttributeError, ValueError):
            # Conservative fallback: rows * cols * 16 bytes
            return int(len(df) * max(1, len(df.columns)) * 16)

    def _streaming_buffer_max_bytes(self) -> int:
        """
        Configurable maximum buffered bytes before flushing to disk.
        """
        mb = self.config.get("streaming_max_buffer_mb", DEFAULT_STREAMING_MAX_BUFFER_MB)
        try:
            mb_i = int(mb)
        except (ValueError, TypeError):
            mb_i = DEFAULT_STREAMING_MAX_BUFFER_MB
        return max(MIN_BUFFER_MB, mb_i) * 1024 * 1024

    def _streaming_flush_max_users(self) -> int:
        """
        Maximum number of per-user chunks to buffer before flushing (independent of bytes).
        """
        v = self.config.get("streaming_flush_max_users", DEFAULT_STREAMING_FLUSH_MAX_USERS)
        try:
            n = int(v)
        except (ValueError, TypeError):
            n = DEFAULT_STREAMING_FLUSH_MAX_USERS
        return max(1, n)

    def _write_csv_append(self, df: pl.DataFrame, *, output_file: Path, include_header: bool) -> None:
        with open(output_file, "ab") as f:
            df.write_csv(f, include_header=include_header)

    def _compute_expected_output_columns(self, database_types: List[str]) -> List[str]:
        """
        Compute a stable CSV schema (final/output column names) for streaming multi-database writes.
        Uses schema `field_categories` keys (standard names) + config display-name mapping.
        """
        field_to_display = CSVFormatConverter.get_field_to_display_name_map()
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        # Strict mode: only output_fields + selected service fields (+ sequence_id)
        if bool(self.config.get("restrict_output_to_config_fields", False)):
            output_fields = CSVFormatConverter.get_output_fields()
            service_allow = self.config.get("service_fields_allowlist")
            service_keep = {str(x) for x in service_allow} if isinstance(service_allow, list) else set()

            cols = [seq_id_col]
            for c in output_fields:
                if c == seq_id_col:
                    continue
                cols.append(field_to_display.get(c, c))
            for c in sorted(service_keep):
                disp = field_to_display.get(c, c)
                if disp not in set(cols):
                    cols.append(disp)
            # De-dupe while preserving order
            seen: set[str] = set()
            out: List[str] = []
            for c in cols:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out

        cols: List[str] = [seq_id_col]
        for f in CSVFormatConverter.get_output_fields():
            if f == seq_id_col:
                continue
            cols.append(field_to_display.get(f, f))

        extra: set[str] = set()
        for db in database_types:
            schema_file = {
                "uom": "uom_schema.json",
                "dexcom": "dexcom_schema.json",
                "libre3": "freestyle_libre3_schema.json",
                "freestyle_libre3": "freestyle_libre3_schema.json",
            }.get(db, f"{db}_schema.json")
            schema_path = Path(__file__).parent / "formats" / schema_file
            if not schema_path.exists():
                continue
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            for standard_name in schema.get("field_categories", {}).keys():
                if standard_name == seq_id_col:
                    continue
                extra.add(field_to_display.get(standard_name, standard_name))

        for c in sorted(extra):
            if c not in set(cols):
                cols.append(c)
        return cols

    def _process_streaming_from_converter(
        self,
        *,
        data_folder: Path,
        database_type: str,
        output_file: Path,
        last_sequence_id: int,
        field_categories_dict: Optional[Dict[str, List[str]]],
    ) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Generic bounded-memory streaming processing for any converter that supports
        `iter_user_event_frames(data_folder, interval_minutes=...)`.
        """
        db_detector = DatabaseDetector()
        output_fields = self.config.get("output_fields")
        database_converter = db_detector.get_database_converter(database_type, self.config or {}, output_fields=output_fields)
        if database_converter is None:
            raise ValueError(f"No converter available for database type: {database_type}")

        iter_fn = getattr(database_converter, "iter_user_event_frames", None)
        if not callable(iter_fn):
            raise ValueError(f"Converter for {database_type} does not support streaming frames")

        output_file.write_text("", encoding="utf-8")

        expected_cols = self._compute_expected_output_columns([database_type])
        max_bytes = self._streaming_buffer_max_bytes()
        max_users = self._streaming_flush_max_users()

        wrote_header = False
        current_last_sequence_id = last_sequence_id
        buffered: List[pl.DataFrame] = []
        buffered_bytes = 0
        buffered_users = 0

        total_records = 0
        total_sequences = 0
        original_records = 0
        min_ts: Optional[str] = None
        max_ts: Optional[str] = None
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        def flush() -> None:
            nonlocal wrote_header, buffered, buffered_bytes, buffered_users, total_records, total_sequences
            if not buffered:
                return
            # Write frames individually to avoid dtype mismatches (Null vs Float64) during concat.
            frames = buffered
            buffered = []
            buffered_bytes = 0
            buffered_users = 0
            for frame in frames:
                missing = [c for c in expected_cols if c not in frame.columns]
                if missing:
                    frame = frame.with_columns([pl.lit(None).alias(c) for c in missing])
                frame = frame.select([c for c in expected_cols if c in frame.columns])
                total_records += len(frame)
                if seq_id_col in frame.columns:
                    total_sequences += int(frame[seq_id_col].n_unique())
                self._write_csv_append(frame, output_file=output_file, include_header=not wrote_header)
                wrote_header = True

        for user_df in iter_fn(data_folder, interval_minutes=self.expected_interval_minutes):
            if len(user_df) == 0:
                continue
            original_records += len(user_df)

            # Track date range
            try:
                umin = user_df[ts_col].min()
                umax = user_df[ts_col].max()
                if umin is not None:
                    s = umin.strftime("%Y-%m-%dT%H:%M:%S")
                    min_ts = s if (min_ts is None or s < min_ts) else min_ts
                if umax is not None:
                    s = umax.strftime("%Y-%m-%dT%H:%M:%S")
                    max_ts = s if (max_ts is None or s > max_ts) else max_ts
            except (KeyError, AttributeError, ValueError):
                pass

            df, _gap_stats, current_last_sequence_id = self.detect_gaps_and_sequences(
                user_df, current_last_sequence_id, field_categories_dict
            )
            df, _ = self.interpolate_missing_values(df, field_categories_dict)
            df, _ = self.filter_sequences_by_length(df)
            if self.create_fixed_frequency:
                df, _ = self.create_fixed_frequency_data(df, field_categories_dict)
            df, _ = self.filter_glucose_only(df)
            ml_df = self.prepare_ml_data(df)

            # Align schema
            missing = [c for c in expected_cols if c not in ml_df.columns]
            if missing:
                ml_df = ml_df.with_columns([pl.lit(None).alias(c) for c in missing])
            ml_df = ml_df.select([c for c in expected_cols if c in ml_df.columns])

            buffered.append(ml_df)
            buffered_bytes += self._df_estimated_size_bytes(ml_df)
            buffered_users += 1

            if buffered_bytes >= max_bytes or buffered_users >= max_users:
                flush()

        flush()

        stats = {
            "dataset_overview": {
                "total_records": total_records,
                "total_sequences": total_sequences,
                "date_range": {"start": min_ts or "N/A", "end": max_ts or "N/A"},
                "original_records": original_records,
            },
            "sequence_analysis": {
                "sequence_lengths": {"count": total_sequences, "mean": 0, "std": 0, "min": 0, "25%": 0, "50%": 0, "75%": 0, "max": 0},
                "longest_sequence": 0,
                "shortest_sequence": 0,
                "sequences_by_length": {},
            },
            "gap_analysis": {},
            "interpolation_analysis": {},
            "calibration_removal_analysis": {},
            "filtering_analysis": {},
            "replacement_analysis": {},
            "fixed_frequency_analysis": {},
            "glucose_filtering_analysis": {},
            "data_quality": {},
        }

        placeholder = pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)})
        return placeholder, stats, current_last_sequence_id
    
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
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
    
    def consolidate_glucose_data(self, data_folder: Path, output_file: Optional[Path] = None) -> pl.DataFrame:
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
        
        logger.info(f"Detected database type: {database_type}")
        
        if database_type == 'unknown':
            raise ValueError(f"Could not detect database type for folder: {data_folder}")
        
        # Get database converter
        database_converter = db_detector.get_database_converter(database_type, self.config or {})
        
        if database_converter is None:
            raise ValueError(f"No converter available for database type: {database_type}")
        
        logger.info(f"Using {database_converter.get_database_name()}")
        
        # Consolidate data using the appropriate converter
        df = database_converter.consolidate_data(data_folder, output_file)
        
        # Store original record count for statistics
        self._original_record_count = len(df)

        return df
        

    def detect_gaps_and_sequences(self, df: pl.DataFrame, last_sequence_id: int = 0, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Detect time gaps and create sequence IDs, marking calibration periods and sequences for removal.
        
        If field_categories_dict is provided, gaps are detected for all continuous fields.
        Sequences are broken if ANY continuous field has a gap > small_gap_max_minutes.
        
        Args:
            df: DataFrame with timestamp column and optionally user_id column
            last_sequence_id: Last sequence ID used (sequences will start from last_sequence_id + 1)
            field_categories_dict: Optional dictionary mapping categories to lists of column names.
                                  If provided, gaps are checked for all continuous fields.
            
        Returns:
            Tuple of (DataFrame with sequence IDs and removal flags, statistics dictionary, last_sequence_id)
        """
        logger.info("Detecting gaps and creating sequences...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        user_id_col = StandardFieldNames.USER_ID
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        # Initialize statistics for calibration period analysis
        calibration_stats = {
            'calibration_periods_detected': 0,
            'sequences_marked_for_removal': 0,  # Not used in current logic but kept for structure
            'total_records_marked_for_removal': 0
        }
        
        # Track current last_sequence_id across user processing
        current_last_sequence_id = last_sequence_id
        
        # Handle multi-user data by processing each user separately
        if user_id_col in df.columns:
            logger.info("Processing multi-user data - creating sequences per user...")
            all_sequences: List[pl.DataFrame] = []
            
            for user_id in sorted(df[user_id_col].unique().to_list()):
                user_data = df.filter(pl.col(user_id_col) == user_id).sort(ts_col)
                user_sequences, user_calib_stats, current_last_sequence_id = self._create_sequences_for_user(
                    user_data, current_last_sequence_id, user_id, field_categories_dict
                )
                all_sequences.append(user_sequences)
                
                # Aggregate stats
                calibration_stats['calibration_periods_detected'] += int(user_calib_stats['calibration_periods_detected'])
                calibration_stats['total_records_marked_for_removal'] += int(user_calib_stats['total_records_marked_for_removal'])
            
            # Combine all user sequences
            if all_sequences:
                df = pl.concat(all_sequences).sort([user_id_col, seq_id_col, ts_col])
            else:
                df = df.clear()
        else:
            # Single user data - process normally
            df = df.sort(ts_col)
            df, calibration_stats, current_last_sequence_id = self._create_sequences_for_user(df, current_last_sequence_id, None, field_categories_dict)
        
        # Calculate statistics
        # Use len() instead of count() for Polars 1.0+ compatibility
        # Handle empty DataFrame or DataFrame without sequence_id column
        if len(df) > 0 and seq_id_col in df.columns:
            if user_id_col in df.columns:
                sequence_counts = df.group_by([user_id_col, seq_id_col]).len().sort([user_id_col, seq_id_col])
            else:
                sequence_counts = df.group_by([seq_id_col]).len().sort(seq_id_col)
            
            stats = {
                'total_sequences': df[seq_id_col].n_unique(),
                'gap_positions': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
                'total_gaps': df['is_gap'].sum() if 'is_gap' in df.columns else 0,
                'sequence_lengths': dict(zip(sequence_counts[seq_id_col].to_list(), sequence_counts['len'].to_list())) if len(sequence_counts) > 0 else {},
                'calibration_period_analysis': calibration_stats
            }
        else:
            # Empty DataFrame or no sequence_id column
            stats = {
                'total_sequences': 0,
                'gap_positions': 0,
                'total_gaps': 0,
                'sequence_lengths': {},
                'calibration_period_analysis': calibration_stats
            }
        
        logger.info(f"Created {stats['total_sequences']} sequences")
        logger.info(f"Found {stats['total_gaps']} gaps > {self.small_gap_max_minutes} minutes")
        
        if calibration_stats['calibration_periods_detected'] > 0:
            logger.info(f"Detected {calibration_stats['calibration_periods_detected']} calibration periods")
            logger.info(f"Removed {calibration_stats['total_records_marked_for_removal']} records after calibration")
        
        # Remove temporary columns
        columns_to_remove = ['time_diff_seconds', 'is_gap', 'is_calibration_gap', 'remove_due_to_calibration']
        df = df.drop([col for col in columns_to_remove if col in df.columns])
        
        return df, stats, current_last_sequence_id
    
    def _create_sequences_for_user(self, user_df: pl.DataFrame, last_sequence_id: int = 0, user_id: Optional[str] = None, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Create sequences for a single user's data and handle calibration periods.
        
        If field_categories_dict is provided, gaps are detected for all continuous fields.
        Sequences are broken if ANY continuous field has a gap > small_gap_max_minutes.
        
        Args:
            user_df: DataFrame with data for a single user
            last_sequence_id: Last sequence ID used (sequences will start from last_sequence_id + 1)
            user_id: User ID (for multi-user data)
            field_categories_dict: Optional dictionary mapping categories to lists of column names.
                                  If provided, gaps are checked for all continuous fields.
            
        Returns:
            Tuple of (DataFrame with sequence IDs added and calibration data removed, calibration statistics, last_sequence_id)
        """
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        stats = {
            'calibration_periods_detected': 0,
            'total_records_marked_for_removal': 0
        }
        
        if len(user_df) == 0:
            return user_df, stats, last_sequence_id

        # Calculate time differences between consecutive timestamps
        df = user_df.with_columns([
            pl.col(ts_col).diff().dt.total_seconds().alias('time_diff_seconds')
        ])
        
        # Identify calibration gaps (based on timestamp differences)
        # A gap is a calibration gap if it exceeds calibration_period_minutes
        df = df.with_columns([
            (pl.col('time_diff_seconds') > self.calibration_period_seconds).fill_null(False).alias('is_calibration_gap')
        ])
        
        # Identify standard gaps (for sequence breaking)
        # Start with timestamp-based gaps (original logic)
        df = df.with_columns([
            (pl.col('time_diff_seconds') > self.small_gap_max_seconds).fill_null(False).alias('is_gap')
        ])
        
        # If field_categories_dict is provided, also check gaps for continuous fields
        if field_categories_dict is not None:
            continuous_fields = field_categories_dict.get('continuous', [])
            continuous_fields = [f for f in continuous_fields if f in df.columns]
            
            continuous_fields_other = [f for f in continuous_fields if f != glucose_col]
            
            if continuous_fields_other:
                continuous_fields_to_check = continuous_fields  # Include glucose
            else:
                continuous_fields_to_check = []
                if glucose_col in df.columns:
                    non_null_rows = df.filter(pl.col(glucose_col).is_not_null()).sort(ts_col)
                    if len(non_null_rows) > 1:
                        non_null_with_diff = non_null_rows.with_columns(
                            pl.col(ts_col).diff().dt.total_seconds().alias('field_time_diff')
                        )
                        gap_rows = non_null_with_diff.filter(
                            pl.col('field_time_diff') > self.small_gap_max_seconds
                        )
                        if len(gap_rows) > 0:
                            gap_timestamps = set(gap_rows[ts_col].to_list())
                            if gap_timestamps:
                                df = df.with_columns(
                                    (
                                        pl.col(glucose_col).is_not_null()
                                        & pl.col(ts_col).is_in(list(gap_timestamps))
                                    ).alias('is_gap_glucose')
                                ).with_columns(
                                    (pl.col('is_gap') | pl.col('is_gap_glucose'))
                                    .fill_null(False)
                                    .alias('is_gap')
                                ).drop('is_gap_glucose')
            
            if continuous_fields_to_check:
                gap_columns: List[pl.Expr] = []
                for field in continuous_fields_to_check:
                    non_null_rows = df.filter(pl.col(field).is_not_null()).sort(ts_col)
                    
                    if len(non_null_rows) > 1:
                        non_null_with_diff = non_null_rows.with_columns([
                            pl.col(ts_col).diff().dt.total_seconds().alias('field_time_diff')
                        ])
                        gap_rows = non_null_with_diff.filter(
                            pl.col('field_time_diff') > self.small_gap_max_seconds
                        )
                        
                        if len(gap_rows) > 0:
                            gap_timestamps = set(gap_rows[ts_col].to_list())
                            field_gap = (
                                pl.col(field).is_not_null() & 
                                pl.col(ts_col).is_in(list(gap_timestamps))
                            )
                        else:
                            field_gap = pl.lit(False)
                    else:
                        field_gap = pl.lit(False)
                    
                    gap_columns.append(field_gap.alias(f'is_gap_{field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}'))
                
                if gap_columns:
                    df = df.with_columns(gap_columns)
                    gap_exprs = [pl.col('is_gap')]
                    for field in continuous_fields_to_check:
                        safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                        gap_exprs.append(pl.col(f'is_gap_{safe_name}'))
                    
                    combined_gap = gap_exprs[0]
                    for expr in gap_exprs[1:]:
                        combined_gap = combined_gap | expr
                    
                    df = df.with_columns([
                        combined_gap.fill_null(False).alias('is_gap')
                    ])
                    
                    temp_gap_cols = [f'is_gap_{field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}' for field in continuous_fields_to_check]
                    df = df.drop([col for col in temp_gap_cols if col in df.columns])
        
        stats['calibration_periods_detected'] = int(df['is_calibration_gap'].sum())
        
        should_remove_calibration = field_categories_dict.get('remove_after_calibration', True) if field_categories_dict else True
        
        if stats['calibration_periods_detected'] > 0 and should_remove_calibration:
            calibration_indices = df.with_row_index().filter(pl.col('is_calibration_gap'))['index'].to_list()
            
            removal_windows: List[Tuple[datetime, datetime]] = []
            timestamps = df[ts_col].to_list()
            
            for idx in calibration_indices:
                start_removal = timestamps[idx]
                end_removal = start_removal + timedelta(hours=self.remove_after_calibration_hours)
                removal_windows.append((start_removal, end_removal))
            
            df = df.with_columns(pl.lit(True).alias('keep_record'))
            for start, end in removal_windows:
                df = df.with_columns(
                    pl.when((pl.col(ts_col) >= start) & (pl.col(ts_col) < end))
                    .then(False)
                    .otherwise(pl.col('keep_record'))
                    .alias('keep_record')
                )
                
            stats['total_records_marked_for_removal'] = int((~df['keep_record']).sum())
            df = df.filter(pl.col('keep_record')).drop('keep_record')
            
        if len(df) > 0:
            df = df.with_columns([
                pl.col(ts_col).diff().dt.total_seconds().alias('time_diff_seconds'),
            ]).with_columns([
                (pl.col('time_diff_seconds') > self.small_gap_max_seconds).fill_null(False).alias('is_gap'),
            ])
            
            if field_categories_dict is not None:
                continuous_fields = field_categories_dict.get('continuous', [])
                continuous_fields = [f for f in continuous_fields if f in df.columns]
                continuous_fields_other = [f for f in continuous_fields if f != glucose_col]
                
                if continuous_fields_other:
                    continuous_fields_to_check = continuous_fields
                else:
                    continuous_fields_to_check = []
                    if glucose_col in df.columns:
                        non_null_rows = df.filter(pl.col(glucose_col).is_not_null()).sort(ts_col)
                        if len(non_null_rows) > 1:
                            non_null_with_diff = non_null_rows.with_columns(
                                pl.col(ts_col).diff().dt.total_seconds().alias('field_time_diff')
                            )
                            gap_rows = non_null_with_diff.filter(
                                pl.col('field_time_diff') > self.small_gap_max_seconds
                            )
                            if len(gap_rows) > 0:
                                gap_timestamps = set(gap_rows[ts_col].to_list())
                                if gap_timestamps:
                                    df = df.with_columns(
                                        (
                                            pl.col(glucose_col).is_not_null()
                                            & pl.col(ts_col).is_in(list(gap_timestamps))
                                        ).alias('is_gap_glucose')
                                    ).with_columns(
                                        (pl.col('is_gap') | pl.col('is_gap_glucose'))
                                        .fill_null(False)
                                        .alias('is_gap')
                                    ).drop('is_gap_glucose')
                
                if continuous_fields_to_check:
                    gap_columns = []
                    for field in continuous_fields_to_check:
                        non_null_rows = df.filter(pl.col(field).is_not_null()).sort(ts_col)
                        if len(non_null_rows) > 1:
                            non_null_with_diff = non_null_rows.with_columns([
                                pl.col(ts_col).diff().dt.total_seconds().alias('field_time_diff')
                            ])
                            gap_rows = non_null_with_diff.filter(
                                pl.col('field_time_diff') > self.small_gap_max_seconds
                            )
                            if len(gap_rows) > 0:
                                gap_timestamps = set(gap_rows[ts_col].to_list())
                                field_gap = (
                                    pl.col(field).is_not_null() & 
                                    pl.col(ts_col).is_in(list(gap_timestamps))
                                )
                            else:
                                field_gap = pl.lit(False)
                        else:
                            field_gap = pl.lit(False)
                        
                        gap_columns.append(field_gap.alias(f'is_gap_{field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}'))
                    
                    if gap_columns:
                        df = df.with_columns(gap_columns)
                        gap_exprs = [pl.col('is_gap')]
                        for field in continuous_fields_to_check:
                            safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                            gap_exprs.append(pl.col(f'is_gap_{safe_name}'))
                        
                        combined_gap = gap_exprs[0]
                        for expr in gap_exprs[1:]:
                            combined_gap = combined_gap | expr
                        
                        df = df.with_columns([
                            combined_gap.fill_null(False).alias('is_gap')
                        ])
                        
                        temp_gap_cols = [f'is_gap_{field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}' for field in continuous_fields_to_check]
                        df = df.drop([col for col in temp_gap_cols if col in df.columns])
            
            df = df.with_columns([
                pl.col('is_gap').cum_sum().alias('local_sequence_id')
            ])
            
            df = df.with_columns([
                (pl.col('local_sequence_id') + last_sequence_id + 1).alias(seq_id_col)
            ]).drop('local_sequence_id')
            
            max_sequence_id = df[seq_id_col].max()
            last_sequence_id = int(max_sequence_id) if max_sequence_id is not None else last_sequence_id
        
        return df, stats, last_sequence_id
    
    def interpolate_missing_values(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, List[str]]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Interpolate only small gaps (1-2 missing data points) within sequences.
        Interpolates glucose values and all fields in the 'continuous' category from field_categories_dict.
        Other columns (occasional, service) are left as empty since they represent discrete events.
        Large gaps are treated as sequence boundaries and not interpolated.
        
        Uses Polars-native operations for better performance.
        
        Args:
            df: DataFrame with sequence IDs and timestamp data
            field_categories_dict: Dictionary mapping categories to lists of column names.
                                  If None, only glucose will be interpolated.
            
        Returns:
            Tuple of (DataFrame with interpolated values, interpolation statistics)
        """
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        event_type_col = StandardFieldNames.EVENT_TYPE
        user_id_col = StandardFieldNames.USER_ID

        # Determine which fields to interpolate
        if field_categories_dict is None:
            # Use standard field name
            field_categories_dict = {
                'continuous': [glucose_col],
                'occasional': [],
                'service': []
            }
        
        continuous_fields = field_categories_dict.get('continuous', [])
        # Always include glucose if it exists
        if glucose_col in df.columns and glucose_col not in continuous_fields:
            continuous_fields.append(glucose_col)
        
        # Filter to only fields that exist in the DataFrame
        fields_to_interpolate = [f for f in continuous_fields if f in df.columns]
        
        if not fields_to_interpolate:
            logger.info("No continuous fields found - skipping interpolation")
            return df, {
                'total_interpolations': 0,
                'total_interpolated_data_points': 0,
                'sequences_processed': 0,
                'small_gaps_filled': 0,
                'large_gaps_skipped': 0
            }
        
        logger.info(f"Interpolating small gaps for fields: {', '.join(fields_to_interpolate)}...")
        
        # Precalculate safe field names (for use in column aliases)
        field_safe_names: Dict[str, str] = {}
        field_stats_keys: Dict[str, str] = {}
        for field in fields_to_interpolate:
            safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            field_safe_names[field] = safe_name
            field_stats_keys[field] = safe_name.lower()
        
        interpolation_stats: Dict[str, Any] = {
            'total_interpolations': 0,
            'total_interpolated_data_points': 0,
            'sequences_processed': 0,
            'small_gaps_filled': 0,
            'large_gaps_skipped': 0
        }
        
        # Add per-field interpolation counts
        for field in fields_to_interpolate:
            interpolation_stats[f'{field_stats_keys[field]}_interpolations'] = 0
        
        interpolation_stats['sequences_processed'] = df[seq_id_col].n_unique()
        
        # Track per-field interpolations for statistics
        per_field_interpolations = {field: 0 for field in fields_to_interpolate}
        
        # Step 1: For each continuous field, fill missing values at existing timestamps
        for field in fields_to_interpolate:
            sequences = df[seq_id_col].unique().to_list()
            
            for seq_id in sequences:
                seq_mask = pl.col(seq_id_col) == seq_id
                seq_df = df.filter(seq_mask).sort(ts_col)
                
                # Filter to rows where this field is not null
                non_null_rows = seq_df.filter(pl.col(field).is_not_null())
                
                if len(non_null_rows) < 2:
                    continue
                
                # Calculate time differences between consecutive non-null values
                non_null_with_diff = non_null_rows.with_columns([
                    (pl.col(ts_col).diff().dt.total_seconds() / 60.0)
                    .alias('time_diff_minutes')
                ])
                
                # Find gaps: time_diff > expected_interval_minutes but <= small_gap_max_minutes
                small_gaps = non_null_with_diff.filter(
                    (pl.col('time_diff_minutes') > self.expected_interval_minutes) &
                    (pl.col('time_diff_minutes') <= self.small_gap_max_minutes)
                )
                
                if small_gaps.height == 0:
                    continue
                
                updates: List[Tuple[datetime, float]] = []
                
                for gap_idx in range(len(small_gaps)):
                    gap_row = small_gaps[gap_idx]
                    curr_timestamp = gap_row[ts_col][0]
                    time_diff = gap_row['time_diff_minutes'][0]
                    curr_value = gap_row[field][0]
                    
                    # Find previous non-null value
                    prev_non_null = non_null_rows.filter(pl.col(ts_col) < curr_timestamp).sort(ts_col, descending=True)
                    if len(prev_non_null) == 0:
                        continue
                    prev_timestamp = prev_non_null[ts_col][0]
                    prev_value = prev_non_null[field][0]
                    
                    if prev_value is None:
                        continue
                    
                    row_before_gap = seq_df.filter(pl.col(ts_col) < curr_timestamp).sort(ts_col, descending=True)
                    if len(row_before_gap) > 0:
                        immediate_prev_value = row_before_gap[field][0]
                        if immediate_prev_value is None:
                            continue
                    
                    # Calculate number of missing points
                    missing_points = int((time_diff / self.expected_interval_minutes) - 1)
                    if missing_points <= 0:
                        continue
                    
                    # Fill missing values at existing timestamps
                    for j in range(1, missing_points + 1):
                        interp_timestamp = prev_timestamp + timedelta(minutes=j * self.expected_interval_minutes)
                        
                        # Check if a row exists at this timestamp and field is missing
                        existing_rows = seq_df.filter(pl.col(ts_col) == interp_timestamp)
                        if existing_rows.height > 0 and existing_rows[field][0] is None:
                            # Interpolate the value
                            alpha = (j * self.expected_interval_minutes) / time_diff
                            interp_value = float(prev_value + alpha * (curr_value - prev_value))
                            updates.append((interp_timestamp, interp_value))
                
                # Apply all updates for this field in this sequence at once
                if updates:
                    per_field_interpolations[field] += len(updates)
                    
                    update_expr = pl.col(field)
                    for ts, val in updates:
                        update_expr = pl.when(
                            (pl.col(seq_id_col) == seq_id) & (pl.col(ts_col) == ts) & (pl.col(field).is_null())
                        ).then(pl.lit(val)).otherwise(update_expr)
                    
                    df = df.with_columns([update_expr.alias(field)])
                    
                    # Update Event Type for interpolated rows
                    if event_type_col in df.columns:
                        event_update_expr = pl.col(event_type_col)
                        for ts, _ in updates:
                            event_update_expr = pl.when(
                                (pl.col(seq_id_col) == seq_id) & (pl.col(ts_col) == ts) & (pl.col(event_type_col) != INTERPOLATED_EVENT_TYPE)
                            ).then(pl.lit(INTERPOLATED_EVENT_TYPE)).otherwise(event_update_expr)
                        df = df.with_columns([event_update_expr.alias(event_type_col)])
        
        # Update statistics for per-field interpolations
        for field in fields_to_interpolate:
            stats_key = field_stats_keys[field]
            interpolation_stats[f'{stats_key}_interpolations'] = per_field_interpolations[field]
        
        # Step 2: Process timestamp-based gaps
        df_with_diffs = df.with_row_index('row_idx').with_columns([
            (pl.col(ts_col).diff().over(seq_id_col).dt.total_seconds() / 60.0).alias('time_diff_minutes')
        ])
        
        df_with_gaps = df_with_diffs.with_columns([
            (
                (pl.col('time_diff_minutes') > self.expected_interval_minutes) &
                (pl.col('time_diff_minutes') <= self.small_gap_max_minutes)
            ).alias('is_small_gap'),
            (pl.col('time_diff_minutes') > self.small_gap_max_minutes).alias('is_large_gap')
        ])
        
        small_gaps_df = df_with_gaps.filter(pl.col('is_small_gap'))
        large_gaps_df = df_with_gaps.filter(pl.col('is_large_gap'))
        
        timestamp_based_gaps = small_gaps_df.height
        interpolation_stats['small_gaps_filled'] = timestamp_based_gaps
        interpolation_stats['large_gaps_skipped'] = large_gaps_df.height
        
        total_interpolations = sum(per_field_interpolations.values())
        interpolation_stats['total_interpolations'] = total_interpolations
        
        if small_gaps_df.height > 0:
            prev_cols = [
                pl.col(ts_col).shift(1).over(seq_id_col).alias('prev_timestamp')
            ]
            
            for field in fields_to_interpolate:
                safe_name = field_safe_names[field]
                prev_cols.append(pl.col(field).shift(1).over(seq_id_col).alias(f'prev_{safe_name}'))
            
            if user_id_col in df_with_gaps.columns:
                prev_cols.append(pl.col(user_id_col).shift(1).over(seq_id_col).alias('prev_user_id'))
            
            df_with_prev = df_with_gaps.with_columns(prev_cols)
            
            gaps_to_process = df_with_prev.filter(pl.col('is_small_gap')).with_columns([
                ((pl.col('time_diff_minutes') / self.expected_interval_minutes).cast(pl.Int64) - 1)
                .alias('missing_points')
            ]).filter(pl.col('missing_points') > 0)
            
            if gaps_to_process.height > 0:
                gaps_with_j = gaps_to_process.with_columns([
                    pl.col('missing_points').map_elements(
                        lambda mp: list(range(1, int(mp) + 1)) if mp and mp > 0 else [],
                        return_dtype=pl.List(pl.Int64)
                    ).alias('j_values')
                ])
                
                gaps_exploded = gaps_with_j.explode('j_values').with_columns([
                    pl.col('j_values').alias('j')
                ])
                
                interpolated_cols = [
                    (pl.col('prev_timestamp') + 
                     pl.duration(minutes=pl.col('j') * self.expected_interval_minutes))
                    .alias(ts_col),
                    ((pl.col('j').cast(pl.Float64) * self.expected_interval_minutes) / 
                     pl.col('time_diff_minutes').cast(pl.Float64))
                    .alias('alpha'),
                    pl.col(seq_id_col),
                ]
                
                for field in fields_to_interpolate:
                    safe_name = field_safe_names[field]
                    interpolated_cols.extend([
                        pl.col(f'prev_{safe_name}'),
                        pl.col(field).alias(f'curr_{safe_name}'),
                    ])
                
                if 'prev_user_id' in gaps_exploded.columns:
                    interpolated_cols.append(pl.col('prev_user_id'))
                
                gaps_calculated = gaps_exploded.select(interpolated_cols)
                
                cast_exprs = []
                for field in fields_to_interpolate:
                    safe_name = field_safe_names[field]
                    cast_exprs.extend([
                        pl.col(f'prev_{safe_name}').cast(pl.Float64, strict=False).alias(f'prev_{safe_name}_num'),
                        pl.col(f'curr_{safe_name}').cast(pl.Float64, strict=False).alias(f'curr_{safe_name}_num'),
                    ])
                
                gaps_calculated = gaps_calculated.with_columns(cast_exprs)
                
                interpolated_field_exprs = []
                for field in fields_to_interpolate:
                    safe_name = field_safe_names[field]
                    prev_col_num = f'prev_{safe_name}_num'
                    curr_col_num = f'curr_{safe_name}_num'
                    
                    interpolated_field_exprs.append(
                        pl.when(
                            (pl.col(prev_col_num).is_not_null()) & 
                            (pl.col(curr_col_num).is_not_null())
                        ).then(
                            pl.col(prev_col_num) + 
                            pl.col('alpha') * (pl.col(curr_col_num) - pl.col(prev_col_num))
                        ).otherwise(None).alias(field)
                    )
                
                interpolated_df = gaps_calculated.with_columns(interpolated_field_exprs)
                
                final_cols = [
                    pl.col(ts_col),
                    pl.col(seq_id_col),
                ]
                
                for field in fields_to_interpolate:
                    final_cols.append(pl.col(field))
                
                if event_type_col in df.columns:
                    final_cols.append(pl.lit(INTERPOLATED_EVENT_TYPE).alias(event_type_col))
                
                if 'prev_user_id' in gaps_calculated.columns:
                    final_cols.append(
                        pl.when(pl.col('prev_user_id').is_not_null())
                        .then(pl.col('prev_user_id'))
                        .otherwise(pl.lit(''))
                        .alias(user_id_col)
                    )
                
                original_schema = df.schema
                existing_col_names = [ts_col, seq_id_col] + fields_to_interpolate
                if event_type_col in df.columns:
                    existing_col_names.append(event_type_col)
                if 'prev_user_id' in gaps_calculated.columns:
                    existing_col_names.append(user_id_col)
                
                for col in df.columns:
                    if col not in existing_col_names:
                        col_type = original_schema[col]
                        is_occasional = col in field_categories_dict.get('occasional', [])
                        is_service = col in field_categories_dict.get('service', [])
                        
                        if is_occasional or is_service:
                            if col_type == pl.Utf8 or col_type == pl.String:
                                final_cols.append(pl.lit('').cast(col_type).alias(col))
                            else:
                                final_cols.append(pl.lit(None).cast(col_type).alias(col))
                        else:
                            if col_type == pl.Utf8 or col_type == pl.String:
                                final_cols.append(pl.lit('').cast(col_type).alias(col))
                            else:
                                final_cols.append(pl.lit(None).cast(col_type).alias(col))
                
                interpolated_df = interpolated_df.select(final_cols)
                interpolated_df = interpolated_df.select(df.columns)
                
                interpolation_stats['total_interpolated_data_points'] = len(interpolated_df)
                
                timestamp_based_interpolations = 0
                for field in fields_to_interpolate:
                    stats_key = field_stats_keys[field]
                    field_interpolations = interpolated_df.filter(pl.col(field).is_not_null()).height
                    interpolation_stats[f'{stats_key}_interpolations'] = interpolation_stats.get(f'{stats_key}_interpolations', 0) + field_interpolations
                    timestamp_based_interpolations += field_interpolations
                
                interpolation_stats['total_interpolations'] = interpolation_stats.get('total_interpolations', 0) + timestamp_based_interpolations
                
                df = pl.concat([df, interpolated_df], how='vertical_relaxed')
        
        if user_id_col in df.columns:
            df = df.sort([user_id_col, seq_id_col, ts_col])
        else:
            df = df.sort([seq_id_col, ts_col])
        
        logger.info(f"Identified and processed {interpolation_stats['small_gaps_filled']} small gaps")
        logger.info(f"Created {interpolation_stats['total_interpolated_data_points']} interpolated data points")
        logger.info(f"Interpolated {interpolation_stats['total_interpolations']} glucose values")
        logger.info(f"Skipped {interpolation_stats['large_gaps_skipped']} large gaps")
        logger.info(f"Processed {interpolation_stats['sequences_processed']} sequences")
        
        return df, interpolation_stats
    
    
    def filter_sequences_by_length(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filter out sequences that are shorter than the minimum required length.
        
        Args:
            df: DataFrame with sequence IDs and processed data
            
        Returns:
            Tuple of (filtered DataFrame, filtering statistics dictionary)
        """
        logger.info(f"Filtering sequences with length < {self.min_sequence_len}...")
        
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        # Calculate sequence lengths
        # Use len() instead of count() for Polars 1.0+ compatibility
        sequence_counts = df.group_by(seq_id_col).len().sort(seq_id_col)
        
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
            logger.info("Warning: No sequences meet the minimum length requirement!")
            return df, filtering_stats
        
        # Filter the DataFrame to keep only sequences that meet the length requirement
        valid_sequence_ids = sequences_to_keep[seq_id_col].to_list()
        filtered_df = df.filter(pl.col(seq_id_col).is_in(valid_sequence_ids))
        
        # Update filtering statistics
        filtering_stats['filtered_records'] = len(filtered_df)
        filtering_stats['removed_records'] = len(df) - len(filtered_df)
        
        logger.info(f"Original sequences: {filtering_stats['original_sequences']}")
        logger.info(f"Sequences after filtering: {filtering_stats['filtered_sequences']}")
        logger.info(f"Sequences removed: {filtering_stats['removed_sequences']}")
        logger.info(f"Original records: {filtering_stats['original_records']:,}")
        logger.info(f"Records after filtering: {filtering_stats['filtered_records']:,}")
        logger.info(f"Records removed: {filtering_stats['removed_records']:,}")
        
        # Show statistics about removed sequences
        if filtering_stats['removed_sequences'] > 0:
            removed_sequences = sequence_counts.filter(pl.col('len') < self.min_sequence_len)
            if len(removed_sequences) > 0:
                min_len_removed = removed_sequences['len'].min()
                max_len_removed = removed_sequences['len'].max()
                avg_len_removed = removed_sequences['len'].mean()
                logger.info(f"Removed sequence lengths - Min: {min_len_removed}, Max: {max_len_removed}, Avg: {avg_len_removed:.1f}")
        
        return filtered_df, filtering_stats
    
    def create_fixed_frequency_data(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, List[str]]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Create fixed-frequency data by aligning sequences to round minutes and ensuring consistent intervals.
        Glucose values are interpolated, while carbs and insulin are shifted to closest datapoints.
        
        Uses a declarative Polars-native approach for better performance and maintainability.
        
        Args:
            df: DataFrame with processed data and sequence IDs
            field_categories_dict: Dictionary mapping categories to lists of column names.
                                  Currently not implemented - added for test compatibility.
                                  Will be used to handle multiple continuous fields in the future.
            
        Returns:
            Tuple of (DataFrame with fixed-frequency data, statistics dictionary)
        """
        logger.info(f"Creating fixed-frequency data with {self.expected_interval_minutes}-minute intervals...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        # Calculate data density metrics BEFORE processing
        before_density_stats = self._calculate_data_density(df, self.expected_interval_minutes)
        
        fixed_freq_stats: Dict[str, Any] = {
            'sequences_processed': 0,
            'total_records_before': len(df),
            'total_records_after': 0,
            'glucose_interpolations': 0,
            'carb_shifted_records': 0,
            'insulin_shifted_records': 0,
            'time_adjustments': 0,
            'data_density_before': before_density_stats,
            'data_density_after': {},
            'density_change_explanation': {}
        }
        
        # Process each sequence using Polars-native operations
        unique_sequences = df[seq_id_col].unique().to_list()
        all_fixed_sequences: List[pl.DataFrame] = []
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col(seq_id_col) == seq_id).sort(ts_col)
            
            if len(seq_data) < 2:
                # Keep single-point sequences as-is, but ensure column order and types match input df
                all_fixed_sequences.append(seq_data.select(df.columns).cast(df.schema))
                continue
                
            fixed_freq_stats['sequences_processed'] += 1
            
            # Create fixed-frequency timestamps using Polars operations
            fixed_seq_data = self._create_fixed_frequency_sequence(seq_data, seq_id, fixed_freq_stats, field_categories_dict)
            
            # Ensure column order and types match input df
            # Use select(df.columns) then cast(df.schema) for perfect alignment
            fixed_seq_data = fixed_seq_data.select(df.columns).cast(df.schema)
            all_fixed_sequences.append(fixed_seq_data)
        
        # Combine all fixed sequences
        if all_fixed_sequences:
            # All sequences now have identical schema due to select() and cast() above
            df_fixed = pl.concat(all_fixed_sequences).sort([seq_id_col, ts_col])
        else:
            df_fixed = df
        
        fixed_freq_stats['total_records_after'] = len(df_fixed)
        
        # Calculate data density metrics AFTER processing
        after_density_stats = self._calculate_data_density(df_fixed, self.expected_interval_minutes)
        fixed_freq_stats['data_density_after'] = after_density_stats
        
        # Calculate percentage of change explained by density change
        density_change_explanation = self._calculate_density_change_explanation(
            fixed_freq_stats['data_density_before'],
            fixed_freq_stats['data_density_after'],
            fixed_freq_stats['total_records_before'],
            fixed_freq_stats['total_records_after']
        )
        fixed_freq_stats['density_change_explanation'] = density_change_explanation
        
        # Print statistics
        logger.info(f"Processed {fixed_freq_stats['sequences_processed']} sequences")
        logger.info(f"Time adjustments made: {fixed_freq_stats['time_adjustments']}")
        logger.info(f"Glucose interpolations: {fixed_freq_stats['glucose_interpolations']}")
        logger.info(f"Insulin records shifted: {fixed_freq_stats['insulin_shifted_records']}")
        logger.info(f"Carb records shifted: {fixed_freq_stats['carb_shifted_records']}")
        logger.info(f"Records before: {fixed_freq_stats['total_records_before']:,}")
        logger.info(f"Records after: {fixed_freq_stats['total_records_after']:,}")
        
        # Print data density and change explanation
        before_density = fixed_freq_stats['data_density_before']
        after_density = fixed_freq_stats['data_density_after']
        explanation = fixed_freq_stats['density_change_explanation']
        
        logger.info(f"Data density: {before_density['avg_points_per_interval']:.2f} -> {after_density['avg_points_per_interval']:.2f} points/interval ({explanation.get('density_change_pct', 0):+.1f}%)")
        logger.info(f"Change explained by density: {explanation.get('explained_pct', 0):.1f}%")
        
        logger.info("Fixed-frequency data creation complete")
        
        return df_fixed, fixed_freq_stats
    
    def _calculate_data_density(self, df: pl.DataFrame, interval_minutes: int) -> Dict[str, Any]:
        """
        Calculate data density metrics: average number of data points per interval.
        Simplified version for performance - only calculates average.
        
        Args:
            df: DataFrame with timestamp column
            interval_minutes: Interval size in minutes
            
        Returns:
            Dictionary with density statistics
        """
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        if len(df) == 0:
            return {
                'avg_points_per_interval': 0.0,
                'total_intervals': 0,
                'total_points': 0
            }
        
        # Group by sequence and calculate intervals per sequence
        interval_seconds = interval_minutes * 60
        total_points = 0
        total_intervals = 0
        
        for seq_id in df[seq_id_col].unique().to_list():
            seq_data = df.filter(pl.col(seq_id_col) == seq_id).sort(ts_col)
            
            if len(seq_data) < 2:
                # Single point sequence - density is 1
                total_points += 1
                total_intervals += 1
                continue
            
            first_ts = seq_data[ts_col].min()
            last_ts = seq_data[ts_col].max()
            if first_ts is None or last_ts is None:
                continue
            duration_seconds = (last_ts - first_ts).total_seconds()
            
            # Calculate number of intervals
            num_intervals = max(1, int(duration_seconds / interval_seconds) + 1)
            num_points = len(seq_data)
            
            total_points += num_points
            total_intervals += num_intervals
        
        return {
            'avg_points_per_interval': total_points / total_intervals if total_intervals > 0 else 0.0,
            'total_intervals': total_intervals,
            'total_points': total_points
        }
    
    def _calculate_density_change_explanation(self, before_density: Dict[str, Any], after_density: Dict[str, Any], 
                                             records_before: int, records_after: int) -> Dict[str, Any]:
        """
        Calculate how much of the record count change can be explained by density change.
        
        Args:
            before_density: Density statistics before processing
            after_density: Density statistics after processing
            records_before: Number of records before
            records_after: Number of records after
            
        Returns:
            Dictionary with explanation statistics
        """
        if before_density['avg_points_per_interval'] == 0:
            return {
                'records_change_pct': 0.0,
                'expected_change_pct': 0.0,
                'explained_pct': 0.0,
                'unexplained_pct': 0.0,
                'expected_records_after': records_before,
                'unexplained_records': 0
            }
        
        # Calculate percentage changes
        records_change_pct = ((records_after - records_before) / records_before * 100) if records_before > 0 else 0.0
        density_change_pct = ((after_density['avg_points_per_interval'] - before_density['avg_points_per_interval']) / 
                              before_density['avg_points_per_interval'] * 100) if before_density['avg_points_per_interval'] > 0 else 0.0
        
        # Expected records after based on density change
        # If density goes from 2.5 to 1.0, we expect records to go from 1000 to 400 (1000 * 1.0/2.5)
        density_ratio = after_density['avg_points_per_interval'] / before_density['avg_points_per_interval'] if before_density['avg_points_per_interval'] > 0 else 1.0
        expected_records_after = int(records_before * density_ratio)
        expected_change_pct = (density_ratio - 1) * 100
        
        # Calculate explained vs unexplained change
        actual_change = records_after - records_before
        expected_change = expected_records_after - records_before
        unexplained_change = actual_change - expected_change
        
        explained_pct = (abs(expected_change) / abs(actual_change) * 100) if actual_change != 0 else 100.0
        unexplained_pct = 100.0 - explained_pct if actual_change != 0 else 0.0
        
        return {
            'records_change_pct': records_change_pct,
            'expected_change_pct': expected_change_pct,
            'explained_pct': explained_pct,
            'unexplained_pct': unexplained_pct,
            'expected_records_after': expected_records_after,
            'unexplained_records': unexplained_change,
            'density_change_pct': density_change_pct
        }
    
    def _create_fixed_frequency_sequence(self, seq_data: pl.DataFrame, seq_id: int, stats: Dict[str, Any], field_categories_dict: Optional[Dict[str, List[str]]] = None) -> pl.DataFrame:
        """
        Create fixed-frequency data for a single sequence using efficient Polars operations.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            stats: Statistics dictionary to update
            field_categories_dict: Dictionary mapping categories to lists of column names.
                                  Used to determine which fields are continuous and need interpolation.
            
        Returns:
            Fixed-frequency sequence as Polars DataFrame
        """
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        event_type_col = StandardFieldNames.EVENT_TYPE
        user_id_col = StandardFieldNames.USER_ID

        # Get first and last timestamps
        first_timestamp = seq_data[ts_col].min()
        last_timestamp = seq_data[ts_col].max()
        if first_timestamp is None or last_timestamp is None:
            return seq_data
        
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
        
        # Create fixed timestamps DataFrame
        fixed_timestamps = pl.DataFrame({
            ts_col: fixed_timestamps_list,
            seq_id_col: [seq_id] * len(fixed_timestamps_list)
        })
        
        # 1. Interpolate Continuous Fields (Linear interpolation)
        service_fields: List[str] = field_categories_dict.get('service', []).copy() if field_categories_dict else []
        if field_categories_dict is None:
            continuous_fields = [glucose_col] if glucose_col in seq_data.columns else []
        else:
            continuous_fields = [f for f in field_categories_dict.get('continuous', []) if f in seq_data.columns]
            # Always include glucose if it exists (safety)
            if glucose_col in seq_data.columns and glucose_col not in continuous_fields:
                continuous_fields.append(glucose_col)
        
        # Interpolate all continuous fields
        result_df = self._interpolate_continuous_fields_linear(fixed_timestamps, seq_data, stats, continuous_fields)
        
        # 2. Shift Events (Nearest Neighbor / Rounding)
        occasional_fields: List[str] = []
        if field_categories_dict is not None:
            occasional_fields = [f for f in field_categories_dict.get('occasional', []) if f in seq_data.columns]
        
        event_cols: List[str] = []
        seen_cols = set()
        result_existing = set(result_df.columns)
        for col in occasional_fields + service_fields + [event_type_col, user_id_col]:
            if col in result_existing:
                continue
            if col in seq_data.columns and col not in continuous_fields and col not in seen_cols:
                event_cols.append(col)
                seen_cols.add(col)
        
        if event_cols:
            numeric_hint: Optional[set[str]] = set(occasional_fields) if occasional_fields else None
            events_df = self._shift_events_rounding(
                seq_data,
                event_cols,
                stats,
                fixed_timestamps_list,
                numeric_cols_hint=numeric_hint,
            )
            # Join events with result (left join to keep all fixed timestamps)
            result_df = result_df.join(events_df, on=ts_col, how='left')
        
        # Ensure all columns from original schema are present
        original_cols = set(seq_data.columns)
        result_cols = set(result_df.columns)
        missing_cols = original_cols - result_cols
        
        if missing_cols:
            for col in missing_cols:
                col_type = seq_data.schema[col]
                if col_type == pl.Utf8 or col_type == pl.String:
                    result_df = result_df.with_columns([pl.lit('').cast(col_type).alias(col)])
                else:
                    result_df = result_df.with_columns([pl.lit(None).cast(col_type).alias(col)])
        
        # Ensure columns are in same order as original
        result_df = result_df.select(seq_data.columns)
        
        return result_df
    
    def _interpolate_continuous_fields_linear(self, fixed_timestamps: pl.DataFrame, seq_data: pl.DataFrame, stats: Dict[str, Any], continuous_fields: List[str]) -> pl.DataFrame:
        """
        Interpolate all continuous fields linearly using previous and next data points.
        
        Args:
            fixed_timestamps: DataFrame with fixed-frequency timestamps
            seq_data: Sequence data with original timestamps and values
            stats: Statistics dictionary to update
            continuous_fields: List of field names to interpolate
            
        Returns:
            DataFrame with all continuous fields interpolated
        """
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE

        if not continuous_fields:
            return fixed_timestamps
        
        result_df = fixed_timestamps
        
        # Interpolate each continuous field
        for field in continuous_fields:
            if field not in seq_data.columns:
                result_df = result_df.with_columns([pl.lit(None).cast(pl.Float64, strict=False).alias(field)])
                continue
            
            seq_data_ts = seq_data.select([ts_col, field]).with_columns([
                pl.col(ts_col).alias('ts_orig'),
                pl.col(field).cast(pl.Float64, strict=False)
            ]).filter(pl.col(field).is_not_null())
            
            if len(seq_data_ts) == 0:
                result_df = result_df.with_columns([pl.lit(None).cast(pl.Float64, strict=False).alias(field)])
                continue
            
            safe_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            
            forward = result_df.join_asof(
                seq_data_ts, 
                on=ts_col, 
                strategy='forward'
            ).rename({field: f'{safe_name}_next', 'ts_orig': f'ts_{safe_name}_next'})
            
            backward = result_df.join_asof(
                seq_data_ts, 
                on=ts_col, 
                strategy='backward'
            ).select([ts_col, field, 'ts_orig']).rename({field: f'{safe_name}_prev', 'ts_orig': f'ts_{safe_name}_prev'})
            
            combined = forward.join(backward.select([ts_col, f'{safe_name}_prev', f'ts_{safe_name}_prev']), on=ts_col, how='left')
            
            combined = combined.with_columns([
                pl.when(pl.col(f'ts_{safe_name}_prev') == pl.col(f'ts_{safe_name}_next')) # Exact match
                .then(pl.col(f'{safe_name}_prev'))
                .when(pl.col(f'ts_{safe_name}_prev').is_null()) # No prev
                .then(pl.col(f'{safe_name}_next'))
                .when(pl.col(f'ts_{safe_name}_next').is_null()) # No next
                .then(pl.col(f'{safe_name}_prev'))
                .otherwise(
                    pl.col(f'{safe_name}_prev') + (
                        (pl.col(f'{safe_name}_next') - pl.col(f'{safe_name}_prev')) * 
                        (pl.col(ts_col) - pl.col(f'ts_{safe_name}_prev')).dt.total_seconds() / 
                        (pl.col(f'ts_{safe_name}_next') - pl.col(f'ts_{safe_name}_prev')).dt.total_seconds()
                    )
                ).alias(field)
            ])
            
            interpolated_field = combined.select([ts_col, field])
            
            if field in result_df.columns:
                result_df = result_df.drop(field).join(
                    interpolated_field,
                    on=ts_col,
                    how='left'
                )
            else:
                result_df = result_df.join(
                    interpolated_field,
                    on=ts_col,
                    how='left'
                )
            
            if field == glucose_col:
                interpolated_count = combined.filter(
                    pl.col(field).is_not_null() & 
                    (pl.col(f'ts_{safe_name}_next') != pl.col(ts_col))
                ).height
                stats['glucose_interpolations'] += interpolated_count
        
        return result_df

    def _shift_events_rounding(
        self,
        seq_data: pl.DataFrame,
        cols: List[str],
        stats: Dict[str, Any],
        fixed_timestamps_list: List[datetime],
        numeric_cols_hint: Optional[set[str]] = None,
    ) -> pl.DataFrame:
        """
        Shift events to nearest grid point from the fixed timestamps list.
        Aggregates multiple events falling into the same bin (sum for numeric, first for string).
        
        Args:
            seq_data: Sequence data with events
            cols: List of event columns to shift
            stats: Statistics dictionary to update
            fixed_timestamps_list: List of fixed timestamps from the grid (must be sorted)
            
        Returns:
            DataFrame with events shifted to nearest grid points
        """
        ts_col = StandardFieldNames.TIMESTAMP
        user_id_col = StandardFieldNames.USER_ID
        event_type_col = StandardFieldNames.EVENT_TYPE

        if len(fixed_timestamps_list) == 0:
            return pl.DataFrame({ts_col: []})
        
        events = seq_data.select([ts_col] + cols)
        events = events.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols]))
        
        if len(events) == 0:
            return pl.DataFrame({ts_col: fixed_timestamps_list[:0]})
        
        fixed_timestamps_df = pl.DataFrame({
            'fixed_timestamp': fixed_timestamps_list
        }).sort('fixed_timestamp')
        
        events_shifted = events.join_asof(
            fixed_timestamps_df,
            left_on=ts_col,
            right_on='fixed_timestamp',
            strategy='nearest'
        ).with_columns([
            pl.col('fixed_timestamp').alias(ts_col)
        ]).drop('fixed_timestamp')
        
        numeric_cols: List[str] = []
        if numeric_cols_hint is not None:
            numeric_cols = [
                c for c in cols
                if c in numeric_cols_hint and c not in {event_type_col, user_id_col} and not c.endswith('_id')
            ]
        else:
            for c in cols:
                if c in {event_type_col, user_id_col} or c.endswith('_id'):
                    continue
                try:
                    any_numeric = seq_data.select(
                        pl.col(c).cast(pl.Float64, strict=False).is_not_null().any()
                    ).item()
                except (ValueError, TypeError, AttributeError):
                    any_numeric = False
                if any_numeric:
                    numeric_cols.append(c)
        cast_exprs = []
        for col in cols:
            if col in numeric_cols:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            else:
                cast_exprs.append(pl.col(col))
        
        if cast_exprs:
            events_shifted = events_shifted.with_columns(cast_exprs)
        
        if 'carb_shifted_records' not in stats:
            stats['carb_shifted_records'] = 0
        if 'insulin_shifted_records' not in stats:
            stats['insulin_shifted_records'] = 0
            
        for col in cols:
            if col in numeric_cols:
                stats['insulin_shifted_records'] += events_shifted.filter(pl.col(col).is_not_null()).height
        
        agg_exprs = []
        for col in cols:
            if col in numeric_cols:
                agg_exprs.append(
                    pl.when(pl.col(col).is_not_null().any())
                    .then(pl.col(col).sum())
                    .otherwise(None)
                    .alias(col)
                )
            else:
                agg_exprs.append(pl.col(col).first().alias(col))
        
        shifted = events_shifted.group_by(ts_col).agg(agg_exprs)

        stabilize_exprs = []
        for c in cols:
            if c not in shifted.columns:
                continue
            if c in numeric_cols:
                stabilize_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
            else:
                stabilize_exprs.append(pl.col(c).cast(pl.Utf8, strict=False).alias(c))
        if stabilize_exprs:
            shifted = shifted.with_columns(stabilize_exprs)
        
        return shifted

    def filter_glucose_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Filter to keep only glucose data with simplified fields.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Tuple of (filtered DataFrame with only glucose data, filtering statistics)
        """
        logger.info("Filtering to glucose-only data...")
        
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        event_type_col = StandardFieldNames.EVENT_TYPE
        fast_acting_col = StandardFieldNames.FAST_ACTING_INSULIN
        long_acting_col = StandardFieldNames.LONG_ACTING_INSULIN
        carb_col = StandardFieldNames.CARB_VALUE

        filtering_stats = {
            'original_records': len(df),
            'glucose_only_enabled': self.glucose_only,
            'records_after_filtering': 0,
            'records_removed': 0,
            'fields_removed': []
        }
        
        if not self.glucose_only:
            logger.info("Glucose-only filtering is disabled - keeping all data")
            filtering_stats['records_after_filtering'] = len(df)
            return df, filtering_stats
        
        # Filter to keep only rows with non-null glucose values - use standard field name
        df_filtered = df.filter(pl.col(glucose_col).is_not_null())
        
        # Remove specified fields - use standard field names
        fields_to_remove = [event_type_col, fast_acting_col, long_acting_col, carb_col]
        existing_fields_to_remove = [field for field in fields_to_remove if field in df_filtered.columns]
        
        if existing_fields_to_remove:
            df_filtered = df_filtered.drop(existing_fields_to_remove)
            filtering_stats['fields_removed'] = existing_fields_to_remove
        
        # Update statistics
        filtering_stats['records_after_filtering'] = len(df_filtered)
        filtering_stats['records_removed'] = len(df) - len(df_filtered)
        
        logger.info(f"Original records: {filtering_stats['original_records']:,}")
        logger.info(f"Records with glucose values: {filtering_stats['records_after_filtering']:,}")
        logger.info(f"Records removed (no glucose): {filtering_stats['records_removed']:,}")
        if filtering_stats['fields_removed']:
            logger.info(f"Fields removed: {', '.join(filtering_stats['fields_removed'])}")
        logger.info("OK: Glucose-only filtering complete")
        
        return df_filtered, filtering_stats
    
    def prepare_ml_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare final DataFrame for machine learning with sequence_id as first column.
        
        Dynamically casts all fields based on field categories and DataFrame schema.
        No hardcoded field names - works with any fields added to config.
        
        Args:
            df: Processed DataFrame with sequence IDs
            
        Returns:
            Final DataFrame ready for ML training
        """
        logger.info("Preparing final ML dataset...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        user_id_col = StandardFieldNames.USER_ID

        cast_exprs: List[pl.Expr] = []
        
        # Get field categories to determine field types dynamically
        field_categories_dict = self._field_categories_dict
        continuous_fields = set(field_categories_dict.get('continuous', [])) if field_categories_dict else set()
        occasional_fields = set(field_categories_dict.get('occasional', [])) if field_categories_dict else set()
        service_fields = set(field_categories_dict.get('service', [])) if field_categories_dict else set()
        
        # Get all fields that should be in output
        output_fields = CSVFormatConverter.get_output_fields()
        all_output_fields = set(output_fields)
        
        # Special fields that always have specific types
        id_fields = {seq_id_col, user_id_col}  # Int64
        
        # Process each column in the DataFrame
        for col in df.columns:
            # Special handling for ID fields
            if col in id_fields:
                cast_exprs.append(pl.col(col).cast(pl.Int64, strict=False).alias(col))
                continue
            
            # Special handling for timestamp
            if col == ts_col:
                if df.schema.get(ts_col) == pl.Datetime:
                    cast_exprs.append(pl.col(ts_col).dt.strftime('%Y-%m-%dT%H:%M:%S').alias(ts_col))
                else:
                    cast_exprs.append(pl.col(ts_col).cast(pl.Utf8, strict=False).alias(ts_col))
                continue
            
            # Determine type based on field categories (priority)
            # Continuous fields are numeric (Float64) - always cast, regardless of current type
            if col in continuous_fields:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            # Occasional fields: numeric if they can be, otherwise string
            elif col in occasional_fields:
                # Check if field can be numeric (many occasional fields like step_count are numeric)
                current_type = df.schema.get(col)
                if current_type in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    # Numeric occasional field - cast to Float64
                    cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
                else:
                    # String occasional field - cast to Utf8
                    cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
            # Service fields are strings (Utf8)
            elif col in service_fields:
                cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
            # If field is in output_fields but not in categories, try to infer type
            elif col in all_output_fields:
                # Infer type from current DataFrame schema
                current_type = df.schema.get(col)
                if current_type in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    # Numeric type - cast to Float64
                    cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
                elif current_type == pl.Datetime:
                    # Datetime - convert to string
                    cast_exprs.append(pl.col(col).dt.strftime('%Y-%m-%dT%H:%M:%S').alias(col))
                else:
                    # Default to string for unknown types
                    cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))
            # For fields not in output_fields but present in DataFrame, preserve their type
            else:
                # Keep original type, but ensure it's a stable type
                current_type = df.schema.get(col)
                if current_type == pl.Datetime:
                    cast_exprs.append(pl.col(col).dt.strftime('%Y-%m-%dT%H:%M:%S').alias(col))
                elif current_type not in [pl.Float64, pl.Int64, pl.Utf8]:
                    # Cast to a stable type if it's not already one
                    if current_type in [pl.Float32, pl.Int32]:
                        cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
                    else:
                        cast_exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(col))

        if cast_exprs:
            df = df.with_columns(cast_exprs)

        # Get column order from config if available, otherwise use default order
        output_fields = CSVFormatConverter.get_output_fields()
        
        # Always include sequence_id and user_id at the beginning if present
        preferred = [seq_id_col] if seq_id_col in df.columns else []
        preferred.extend([f for f in output_fields if f != seq_id_col])
        if user_id_col in df.columns and user_id_col not in preferred:
            preferred.append(user_id_col)
        
        # Order columns: preferred fields first, then any remaining columns
        ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in set(preferred)]
        ml_df = df.select(ordered)
        
        # Convert standard field names to display names for CSV output
        field_to_display_map = CSVFormatConverter.get_field_to_display_name_map()
        
        # Rename columns: use display name if mapping exists, otherwise use standard name
        rename_map: Dict[str, str] = {}
        for col in ml_df.columns:
            if col in field_to_display_map:
                rename_map[col] = field_to_display_map[col]
        
        if rename_map:
            ml_df = ml_df.rename(rename_map)

        # Optional strict output filtering:
        if bool(self.config.get("restrict_output_to_config_fields", False)):
            service_allow = self.config.get("service_fields_allowlist")
            if isinstance(service_allow, list):
                service_keep = {str(x) for x in service_allow}
            else:
                service_keep = set(service_fields)

            allowed_standard = set(output_fields) | service_keep | {seq_id_col}
            if user_id_col in df.columns:
                allowed_standard.add(user_id_col)

            allowed_cols = {field_to_display_map.get(c, c) for c in allowed_standard}

            if seq_id_col in ml_df.columns:
                allowed_cols.add(seq_id_col)

            ml_df = ml_df.select([c for c in ml_df.columns if c in allowed_cols])
        
        return ml_df
    
    def get_statistics(self, df: pl.DataFrame, gap_stats: Dict[str, Any], interp_stats: Dict[str, Any], removal_stats: Optional[Dict[str, Any]] = None, filter_stats: Optional[Dict[str, Any]] = None, replacement_stats: Optional[Dict[str, Any]] = None, glucose_filter_stats: Optional[Dict[str, Any]] = None, fixed_freq_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        event_type_col = StandardFieldNames.EVENT_TYPE
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        fast_insulin_col = StandardFieldNames.FAST_ACTING_INSULIN
        long_insulin_col = StandardFieldNames.LONG_ACTING_INSULIN
        carb_col = StandardFieldNames.CARB_VALUE

        # Get date range from timestamp column if available (supports Datetime or string timestamps)
        date_range = {'start': 'N/A', 'end': 'N/A'}
        if ts_col in df.columns:
            ts_dtype = df.schema.get(ts_col)
            valid_timestamps = df.filter(pl.col(ts_col).is_not_null())
            if len(valid_timestamps) > 0:
                if ts_dtype == pl.Datetime:
                    timestamps = valid_timestamps[ts_col].dt.strftime('%Y-%m-%dT%H:%M:%S').sort()
                else:
                    timestamps = valid_timestamps[ts_col].cast(pl.Utf8, strict=False).sort()
                date_range = {'start': timestamps[0], 'end': timestamps[-1]}
        
        # Calculate sequence statistics
        # Use len() instead of count() for Polars 1.0+ compatibility
        sequence_counts = df.group_by(seq_id_col).len().sort(seq_id_col)
        seq_lens = sequence_counts['len']
        
        sequence_lengths_stats = {
            'count': len(seq_lens),
            'mean': seq_lens.mean() if not seq_lens.is_empty() else 0,
            'std': seq_lens.std() if not seq_lens.is_empty() else 0,
            'min': seq_lens.min() if not seq_lens.is_empty() else 0,
            '25%': seq_lens.quantile(0.25) if not seq_lens.is_empty() else 0,
            '50%': seq_lens.median() if not seq_lens.is_empty() else 0,
            '75%': seq_lens.quantile(0.75) if not seq_lens.is_empty() else 0,
            'max': seq_lens.max() if not seq_lens.is_empty() else 0
        }
        
        # Calculate sequences by length
        if not seq_lens.is_empty():
            counts_df = seq_lens.value_counts().sort("len")
            sequences_by_length = dict(zip(counts_df["len"].to_list(), counts_df["count"].to_list()))
        else:
            sequences_by_length = {}

        stats = {
            'dataset_overview': {
                'total_records': len(df),
                'total_sequences': df[seq_id_col].n_unique() if seq_id_col in df.columns else 0,
                'date_range': date_range,
                'original_records': getattr(self, '_original_record_count', len(df))
            },
            'sequence_analysis': {
                'sequence_lengths': sequence_lengths_stats,
                'longest_sequence': sequence_lengths_stats['max'],
                'shortest_sequence': sequence_lengths_stats['min'],
                'sequences_by_length': sequences_by_length
            },
            'gap_analysis': gap_stats,
            'interpolation_analysis': interp_stats,
            'calibration_removal_analysis': removal_stats if removal_stats else {},
            'filtering_analysis': filter_stats if filter_stats else {},
            'replacement_analysis': replacement_stats if replacement_stats else {},
            'fixed_frequency_analysis': fixed_freq_stats if fixed_freq_stats else {},
            'glucose_filtering_analysis': glucose_filter_stats if glucose_filter_stats else {},
            'data_quality': {}
        }
        
        # Build data_quality section using standard field names
        stats['data_quality'] = {
            'glucose_data_completeness': (1 - df[glucose_col].null_count() / len(df)) * 100 if glucose_col in df.columns and len(df) > 0 else 0,
            'fast_acting_insulin_data_completeness': (1 - df[fast_insulin_col].null_count() / len(df)) * 100 if fast_insulin_col in df.columns and len(df) > 0 else 0,
            'long_acting_insulin_data_completeness': (1 - df[long_insulin_col].null_count() / len(df)) * 100 if long_insulin_col in df.columns and len(df) > 0 else 0,
            'carb_data_completeness': (1 - df[carb_col].null_count() / len(df)) * 100 if carb_col in df.columns and len(df) > 0 else 0,
            'interpolated_records': df.filter(pl.col(event_type_col) == INTERPOLATED_EVENT_TYPE).height if event_type_col in df.columns else 0
        }
        
        return stats
    
    def process(self, csv_folder: Path, output_file: Optional[Path] = None, last_sequence_id: int = 0) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Complete preprocessing pipeline with mandatory consolidation.
        
        Args:
            csv_folder: Path to folder containing CSV files (consolidation is mandatory)
            output_file: Optional path to save processed data
            last_sequence_id: Last sequence ID used (sequences will start from last_sequence_id + 1)
            
        Returns:
            Tuple of (processed DataFrame, statistics dictionary, last_sequence_id)
        """
        logger.info("Starting glucose data preprocessing for ML...")
        logger.info(f"Time discretization interval: {self.expected_interval_minutes} minutes")
        logger.info(f"Small gap max (interpolation limit): {self.small_gap_max_minutes} minutes")
        logger.info(f"Save intermediate files: {self.save_intermediate_files}")
        logger.info("-" * 50)
        
        # Detect database type to extract field categories for interpolation
        db_detector = DatabaseDetector()
        database_type = db_detector.detect_database_type(csv_folder)
        field_categories_dict = self.extract_field_categories(database_type) if database_type != 'unknown' else None
        self._field_categories_dict = field_categories_dict
        
        # Generic bounded-memory streaming
        database_converter = db_detector.get_database_converter(database_type, self.config or {})
        if (
            output_file
            and database_converter is not None
            and callable(getattr(database_converter, "iter_user_event_frames", None))
        ):
            return self._process_streaming_from_converter(
                data_folder=csv_folder,
                database_type=database_type,
                output_file=output_file,
                last_sequence_id=last_sequence_id,
                field_categories_dict=field_categories_dict,
            )
        
        # Step 1: Consolidate CSV files (mandatory step)
        consolidated_file = Path("consolidated_data.csv") if self.save_intermediate_files else None
        
        logger.info("STEP 1: Consolidating CSV files (mandatory step)...")
        df = self.consolidate_glucose_data(csv_folder, consolidated_file)
        
        if self.save_intermediate_files:
            logger.info(f"Consolidated data saved to: {consolidated_file}")
        
        logger.info("-" * 40)
        
        # Step 2: Detect gaps and create sequences
        logger.info("STEP 2: Detecting gaps and creating sequences...")
        df, gap_stats, last_sequence_id = self.detect_gaps_and_sequences(df, last_sequence_id, field_categories_dict)
        logger.info("OK: Gap detection and sequence creation complete")
        
        if self.save_intermediate_files:
            intermediate_file = Path("step2_sequences_created.csv")
            df.write_csv(intermediate_file)
            logger.info(f"Data with sequences saved to: {intermediate_file}")
        
        logger.info("-" * 40)
        
        # Step 3: Interpolate missing values
        logger.info("STEP 3: Interpolating missing values...")
        df, interp_stats = self.interpolate_missing_values(df, field_categories_dict)
        logger.info("OK: Missing value interpolation complete")
        
        if self.save_intermediate_files:
            intermediate_file = Path("step3_interpolated_values.csv")
            df.write_csv(intermediate_file)
            logger.info(f"Data with interpolated values saved to: {intermediate_file}")
        
        logger.info("-" * 40)
        
        # Step 4: Filter sequences by minimum length
        logger.info("STEP 4: Filtering sequences by minimum length...")
        df, filter_stats = self.filter_sequences_by_length(df)
        logger.info("OK: Sequence filtering complete")
        
        if self.save_intermediate_files:
            intermediate_file = Path("step4_filtered_sequences.csv")
            df.write_csv(intermediate_file)
            logger.info(f"Filtered data saved to: {intermediate_file}")
        
        logger.info("-" * 40)
        
        # Step 5: Create fixed-frequency data (if enabled)
        if self.create_fixed_frequency:
            logger.info("STEP 5: Creating fixed-frequency data...")
            df, fixed_freq_stats = self.create_fixed_frequency_data(df, field_categories_dict)
            logger.info("Fixed-frequency data creation complete")
            
            if self.save_intermediate_files:
                intermediate_file = Path("step5_fixed_frequency.csv")
                df.write_csv(intermediate_file)
                logger.info(f"Fixed-frequency data saved to: {intermediate_file}")
        else:
            logger.info("STEP 5: Fixed-frequency data creation is disabled - skipping")
            fixed_freq_stats = {}
        
        logger.info("-" * 40)
        
        # Step 6: Filter to glucose-only data (if requested)
        logger.info("STEP 6: Filtering to glucose-only data...")
        df, glucose_filter_stats = self.filter_glucose_only(df)
        logger.info("OK: Glucose-only filtering complete")
        
        if self.save_intermediate_files:
            intermediate_file = Path("step6_glucose_only.csv")
            df.write_csv(intermediate_file)
            logger.info(f"Glucose-only data saved to: {intermediate_file}")
        
        logger.info("-" * 40)
        
        # Step 7: Prepare final ML dataset
        logger.info("STEP 7: Preparing final ML dataset...")
        ml_df = self.prepare_ml_data(df)
        logger.info("OK: ML dataset preparation complete")
        
        if self.save_intermediate_files:
            intermediate_file = Path("step7_ml_ready.csv")
            ml_df.write_csv(intermediate_file)
            logger.info(f"ML-ready data saved to: {intermediate_file}")
        
        logger.info("-" * 40)
        
        # Generate statistics
        stats = self.get_statistics(ml_df, gap_stats, interp_stats, None, filter_stats, None, glucose_filter_stats, fixed_freq_stats)
        
        # Save final output if specified
        if output_file:
            ml_df.write_csv(output_file)
            logger.info(f"Final processed data saved to: {output_file}")
        
        logger.info("-" * 50)
        logger.info("Preprocessing completed successfully!")
        
        return ml_df, stats, last_sequence_id

    def process_multiple_databases(self, csv_folders: List[Path], output_file: Optional[Path] = None, last_sequence_id: int = 0) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Process multiple databases with different formats and combine them into a single output.
        Sequence IDs are tracked consistently across databases using last_sequence_id parameter.
        
        Args:
            csv_folders: List of paths to folders containing CSV files
            output_file: Optional path to save combined processed data
            last_sequence_id: Last sequence ID used (sequences will start from last_sequence_id + 1)
            
        Returns:
            Tuple of (combined DataFrame, aggregated statistics dictionary, last_sequence_id)
        """
        logger.info(f"Starting multi-database processing for {len(csv_folders)} databases...")
        logger.info(f"Databases to process: {', '.join(str(p) for p in csv_folders)}")
        logger.info("-" * 50)

        db_detector = DatabaseDetector()
        db_types: List[str] = []
        for p in csv_folders:
            try:
                db_types.append(db_detector.detect_database_type(p))
            except (ValueError, OSError):
                db_types.append("unknown")
        
        converters = [db_detector.get_database_converter(t, self.config or {}) for t in db_types]
        streaming_capable = any(c is not None and callable(getattr(c, "iter_user_event_frames", None)) for c in converters)
        
        ts_col = StandardFieldNames.TIMESTAMP
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        user_id_col = StandardFieldNames.USER_ID

        if streaming_capable and output_file:
            expected_cols = self._compute_expected_output_columns(db_types)
            max_bytes = self._streaming_buffer_max_bytes()
            max_users = self._streaming_flush_max_users()
            output_file.write_text("", encoding="utf-8")

            current_last_sequence_id = last_sequence_id
            wrote_header = False
            buffered: List[pl.DataFrame] = []
            buffered_bytes = 0
            buffered_users = 0
            total_records = 0
            total_sequences = 0
            min_ts: Optional[str] = None
            max_ts: Optional[str] = None

            def flush() -> None:
                nonlocal wrote_header, buffered, buffered_bytes, buffered_users, total_records, total_sequences
                if not buffered:
                    return
                frames = buffered
                buffered = []
                buffered_bytes = 0
                buffered_users = 0
                for frame in frames:
                    missing = [c for c in expected_cols if c not in frame.columns]
                    if missing:
                        frame = frame.with_columns([pl.lit(None).alias(c) for c in missing])
                    frame = frame.select([c for c in expected_cols if c in frame.columns])
                    total_records += len(frame)
                    if seq_id_col in frame.columns:
                        total_sequences += int(frame[seq_id_col].n_unique())
                    self._write_csv_append(frame, output_file=output_file, include_header=not wrote_header)
                    wrote_header = True

            for idx, (csv_folder, db_type, converter) in enumerate(zip(csv_folders, db_types, converters), 1):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"PROCESSING DATABASE {idx}/{len(csv_folders)}: {csv_folder}")
                logger.info(f"{'=' * 60}\n")

                if converter is not None and callable(getattr(converter, "iter_user_event_frames", None)):
                    field_categories_dict = self.extract_field_categories(db_type) if db_type != "unknown" else None
                    for user_df in converter.iter_user_event_frames(csv_folder, interval_minutes=self.expected_interval_minutes):
                        if len(user_df) == 0:
                            continue

                        try:
                            umin = user_df[ts_col].min()
                            umax = user_df[ts_col].max()
                            if umin is not None:
                                s = umin.strftime("%Y-%m-%dT%H:%M:%S")
                                min_ts = s if (min_ts is None or s < min_ts) else min_ts
                            if umax is not None:
                                s = umax.strftime("%Y-%m-%dT%H:%M:%S")
                                max_ts = s if (max_ts is None or s > max_ts) else max_ts
                        except (KeyError, AttributeError, ValueError):
                            pass

                        df, _gap_stats, current_last_sequence_id = self.detect_gaps_and_sequences(
                            user_df, current_last_sequence_id, field_categories_dict
                        )
                        df, _ = self.interpolate_missing_values(df, field_categories_dict)
                        df, _ = self.filter_sequences_by_length(df)
                        if self.create_fixed_frequency:
                            df, _ = self.create_fixed_frequency_data(df, field_categories_dict)
                        df, _ = self.filter_glucose_only(df)
                        ml_df = self.prepare_ml_data(df)

                        missing = [c for c in expected_cols if c not in ml_df.columns]
                        if missing:
                            ml_df = ml_df.with_columns([pl.lit(None).alias(c) for c in missing])
                        ml_df = ml_df.select([c for c in expected_cols if c in ml_df.columns])

                        buffered.append(ml_df)
                        buffered_bytes += self._df_estimated_size_bytes(ml_df)
                        buffered_users += 1
                        if buffered_bytes >= max_bytes or buffered_users >= max_users:
                            flush()
                else:
                    ml_df, stats, current_last_sequence_id = self.process(
                        csv_folder, output_file=None, last_sequence_id=current_last_sequence_id
                    )
                    try:
                        dr = stats.get("dataset_overview", {}).get("date_range", {}) if isinstance(stats, dict) else {}
                        if dr.get("start") and dr["start"] != "N/A":
                            min_ts = dr["start"] if (min_ts is None or dr["start"] < min_ts) else min_ts
                        if dr.get("end") and dr["end"] != "N/A":
                            max_ts = dr["end"] if (max_ts is None or dr["end"] > max_ts) else max_ts
                    except (KeyError, AttributeError, ValueError):
                        pass

                    missing = [c for c in expected_cols if c not in ml_df.columns]
                    if missing:
                        ml_df = ml_df.with_columns([pl.lit(None).alias(c) for c in missing])
                    ml_df = ml_df.select([c for c in expected_cols if c in ml_df.columns])

                    buffered.append(ml_df)
                    buffered_bytes += self._df_estimated_size_bytes(ml_df)
                    buffered_users += 1
                    if buffered_bytes >= max_bytes or buffered_users >= max_users:
                        flush()

            flush()

            aggregated_stats = {
                "dataset_overview": {
                    "total_records": total_records,
                    "total_sequences": total_sequences,
                    "date_range": {"start": min_ts or "N/A", "end": max_ts or "N/A"},
                    "original_records": total_records,
                },
                "sequence_analysis": {
                    "sequence_lengths": {"count": total_sequences, "mean": 0, "std": 0, "min": 0, "25%": 0, "50%": 0, "75%": 0, "max": 0},
                    "longest_sequence": 0,
                    "shortest_sequence": 0,
                    "sequences_by_length": {},
                },
                "gap_analysis": {},
                "interpolation_analysis": {},
                "calibration_removal_analysis": {},
                "filtering_analysis": {},
                "replacement_analysis": {},
                "fixed_frequency_analysis": {},
                "glucose_filtering_analysis": {},
                "data_quality": {},
            }

            placeholder = pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)})
            return placeholder, aggregated_stats, current_last_sequence_id
        
        all_dataframes: List[pl.DataFrame] = []
        all_statistics: List[Dict[str, Any]] = []
        current_last_sequence_id = last_sequence_id
        
        for idx, csv_folder in enumerate(csv_folders, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"PROCESSING DATABASE {idx}/{len(csv_folders)}: {csv_folder}")
            logger.info(f"{'=' * 60}\n")
            
            ml_df, stats, current_last_sequence_id = self.process(csv_folder, output_file=None, last_sequence_id=current_last_sequence_id)
            
            output_fields = CSVFormatConverter.get_output_fields()
            if user_id_col in ml_df.columns and user_id_col not in output_fields:
                logger.info(f"\nRemoving user_id column for multi-database compatibility...")
                ml_df = ml_df.drop(user_id_col)
            
            max_seq_id = ml_df[seq_id_col].max() if len(ml_df) > 0 else current_last_sequence_id
            min_seq_id = ml_df[seq_id_col].min() if len(ml_df) > 0 else current_last_sequence_id
            
            stats['database_info'] = {
                'database_index': idx,
                'database_path': str(csv_folder),
                'sequence_id_start': current_last_sequence_id + 1 if idx == 1 else None,
                'sequence_id_range': {
                    'min': int(min_seq_id) if min_seq_id is not None else current_last_sequence_id,
                    'max': int(max_seq_id) if max_seq_id is not None else current_last_sequence_id
                }
            }
            
            all_dataframes.append(ml_df)
            all_statistics.append(stats)
            
            logger.info(f"\nDatabase {idx} processed: {len(ml_df):,} records, {ml_df[seq_id_col].n_unique():,} sequences")
            logger.info(f"   Sequence ID range: {min_seq_id} - {max_seq_id}")
            logger.info(f"   Last sequence ID after processing: {current_last_sequence_id}")
        
        logger.info(f"\n{'=' * 60}")
        logger.info("COMBINING ALL DATABASES")
        logger.info(f"{'=' * 60}\n")
        
        combined_df = pl.concat(all_dataframes)
        combined_df = combined_df.sort([seq_id_col, ts_col])
        
        combined_stats = self._aggregate_statistics(all_statistics, csv_folders)
        
        logger.info(f"Combined {len(csv_folders)} databases:")
        logger.info(f"   Total records: {len(combined_df):,}")
        logger.info(f"   Total sequences: {combined_df[seq_id_col].n_unique():,}")
        logger.info(f"   Sequence ID range: {combined_df[seq_id_col].min()} - {combined_df[seq_id_col].max()}")
        
        if output_file:
            combined_df.write_csv(output_file)
            logger.info(f"\nFinal combined data saved to: {output_file}")
        
        logger.info("-" * 50)
        logger.info(f"Multi-database preprocessing completed successfully!")
        logger.info(f"Final last sequence ID: {current_last_sequence_id}")
        
        return combined_df, combined_stats, current_last_sequence_id
    
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
                'glucose_value_mgdl_interpolations': 0,
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
            # Handle both old format (with slash) and new format (without slash)
            glucose_interps = interp_analysis.get('glucose_value_mgdl_interpolations', 
                                                  interp_analysis.get('glucose_value_mg/dl_interpolations', 0))
            aggregated['interpolation_analysis']['glucose_value_mgdl_interpolations'] += glucose_interps
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
                
                # Aggregate data density (weighted average)
                if 'data_density_before' in fixed_freq_analysis and 'data_density_after' in fixed_freq_analysis:
                    before_density = fixed_freq_analysis['data_density_before']
                    after_density = fixed_freq_analysis['data_density_after']
                    
                    if 'data_density_before' not in aggregated['fixed_frequency_analysis']:
                        aggregated['fixed_frequency_analysis']['data_density_before'] = {
                            'total_points': 0,
                            'total_intervals': 0
                        }
                    if 'data_density_after' not in aggregated['fixed_frequency_analysis']:
                        aggregated['fixed_frequency_analysis']['data_density_after'] = {
                            'total_points': 0,
                            'total_intervals': 0
                        }
                    
                    agg_before = aggregated['fixed_frequency_analysis']['data_density_before']
                    agg_after = aggregated['fixed_frequency_analysis']['data_density_after']
                    
                    agg_before['total_points'] += before_density.get('total_points', 0)
                    agg_before['total_intervals'] += before_density.get('total_intervals', 0)
                    
                    agg_after['total_points'] += after_density.get('total_points', 0)
                    agg_after['total_intervals'] += after_density.get('total_intervals', 0)
        
        # Recalculate aggregated density metrics
        if 'fixed_frequency_analysis' in aggregated and 'data_density_before' in aggregated['fixed_frequency_analysis']:
            before_density = aggregated['fixed_frequency_analysis']['data_density_before']
            after_density = aggregated['fixed_frequency_analysis']['data_density_after']
            
            # Calculate final averages
            if before_density.get('total_intervals', 0) > 0:
                before_density['avg_points_per_interval'] = before_density['total_points'] / before_density['total_intervals']
            
            if after_density.get('total_intervals', 0) > 0:
                after_density['avg_points_per_interval'] = after_density['total_points'] / after_density['total_intervals']
            
            # Recalculate density change explanation for aggregated data
            if before_density.get('total_intervals', 0) > 0 and after_density.get('total_intervals', 0) > 0:
                aggregated['fixed_frequency_analysis']['density_change_explanation'] = self._calculate_density_change_explanation(
                    before_density,
                    after_density,
                    aggregated['fixed_frequency_analysis'].get('total_records_before', 0),
                    aggregated['fixed_frequency_analysis'].get('total_records_after', 0)
                )
        
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
    logger.info("\n" + "="*60)
    logger.info("GLUCOSE DATA PREPROCESSING STATISTICS")
    logger.info("="*60)
    
    # Show multi-database information if present
    if 'multi_database_info' in stats:
        multi_db_info = stats['multi_database_info']
        logger.info(f"\nMULTI-DATABASE PROCESSING:")
        logger.info(f"   Total Databases Combined: {multi_db_info['total_databases']}")
        logger.info(f"   Database Paths:")
        for i, path in enumerate(multi_db_info['database_paths'], 1):
            logger.info(f"      {i}. {path}")
        
        logger.info(f"\n   Processed Databases Details:")
        for db in multi_db_info['databases_processed']:
            db_idx = db.get('database_index', 'N/A')
            db_name = db.get('database_name', 'Unknown')
            seq_range = db.get('sequence_id_range', {})
            logger.info(f"      Database {db_idx} ({db_name}):")
            logger.info(f"         Sequence ID Range: {seq_range.get('min', 'N/A')} - {seq_range.get('max', 'N/A')}")
    
    # Show parameters if preprocessor is provided
    if preprocessor:
        logger.info(f"\nPARAMETERS USED:")
        logger.info(f"   Time Discretization Interval: {preprocessor.expected_interval_minutes} minutes")
        logger.info(f"   Small Gap Max (Interpolation Limit): {preprocessor.small_gap_max_minutes} minutes")
        logger.info(f"   Remove Calibration Events: {preprocessor.remove_calibration}")
        logger.info(f"   Minimum Sequence Length: {preprocessor.min_sequence_len}")
        logger.info(f"   Calibration Period Threshold: {preprocessor.calibration_period_minutes} minutes")
        logger.info(f"   Remove After Calibration: {preprocessor.remove_after_calibration_hours} hours")
        logger.info(f"   Create Fixed-Frequency Data: {preprocessor.create_fixed_frequency}")
    
    # Dataset Overview
    overview = stats['dataset_overview']
    logger.info(f"\nDATASET OVERVIEW:")
    logger.info(f"   Total Records: {overview['total_records']:,}")
    logger.info(f"   Total Sequences: {overview['total_sequences']:,}")
    logger.info(f"   Date Range: {overview['date_range']['start']} to {overview['date_range']['end']}")
    
    # Show data preservation percentage
    original_records = overview.get('original_records', overview['total_records'])
    final_records = overview['total_records']
    preservation_percentage = (final_records / original_records * 100) if original_records > 0 else 100
    logger.info(f"   Data Preservation: {preservation_percentage:.1f}% ({final_records:,}/{original_records:,} records)")
    
    # Sequence Analysis
    seq_analysis = stats['sequence_analysis']
    logger.info(f"\nSEQUENCE ANALYSIS:")
    logger.info(f"   Longest Sequence: {seq_analysis['longest_sequence']:,} records")
    logger.info(f"   Shortest Sequence: {seq_analysis['shortest_sequence']:,} records")
    logger.info(f"   Average Sequence Length: {seq_analysis['sequence_lengths']['mean']:.1f} records")
    logger.info(f"   Median Sequence Length: {seq_analysis['sequence_lengths']['50%']:.1f} records")
    
    # Gap Analysis
    gap_analysis = stats.get('gap_analysis', {})
    if gap_analysis:
        logger.info(f"\nGAP ANALYSIS:")
        logger.info(f"   Total Gaps > {preprocessor.small_gap_max_minutes if preprocessor else 'N/A'} minutes: {gap_analysis.get('total_gaps', 0):,}")
        logger.info(f"   Sequences Created: {gap_analysis.get('total_sequences', 0):,}")
    
    # Calibration Period Analysis
    if 'calibration_period_analysis' in gap_analysis and gap_analysis['calibration_period_analysis']:
        calib_analysis = gap_analysis['calibration_period_analysis']
        logger.info(f"\nCALIBRATION PERIOD ANALYSIS:")
        logger.info(f"   Calibration Periods Detected: {calib_analysis.get('calibration_periods_detected', 0):,}")
        logger.info(f"   Records Removed After Calibration: {calib_analysis.get('total_records_marked_for_removal', 0):,}")
        logger.info(f"   Sequences Affected: {calib_analysis.get('sequences_marked_for_removal', 0):,}")
    
    # High/Low Value Replacement Analysis
    if 'replacement_analysis' in stats and stats['replacement_analysis']:
        replacement_analysis = stats['replacement_analysis']
        logger.info(f"\nHIGH/LOW VALUE REPLACEMENT ANALYSIS:")
        logger.info(f"   High Values Replaced (-> 401): {replacement_analysis['high_replacements']:,}")
        logger.info(f"   Low Values Replaced (-> 39): {replacement_analysis['low_replacements']:,}")
        logger.info(f"   Total Replacements: {replacement_analysis['total_replacements']:,}")
        logger.info(f"   Glucose Field Type: {'Float64' if replacement_analysis['glucose_field_converted_to_float'] else 'String'}")
    
    # Interpolation Analysis
    interp_analysis = stats.get('interpolation_analysis', {})
    if interp_analysis:
        logger.info(f"\nINTERPOLATION ANALYSIS:")
        logger.info(f"   Small Gaps Identified and Processed: {interp_analysis.get('small_gaps_filled', 0):,}")
        logger.info(f"   Interpolated Data Points Created: {interp_analysis.get('total_interpolated_data_points', 0):,}")
        logger.info(f"   Total Field Interpolations: {interp_analysis.get('total_interpolations', 0):,}")
        # Handle both old format (with slash) and new format (without slash)
        glucose_interps = interp_analysis.get('glucose_value_mgdl_interpolations', 
                                              interp_analysis.get('glucose_value_mg/dl_interpolations', 0))
        logger.info(f"   Glucose Interpolations: {glucose_interps:,}")
        logger.info(f"   Insulin Interpolations: {interp_analysis.get('insulin_value_u_interpolations', 0):,}")
        logger.info(f"   Carb Interpolations: {interp_analysis.get('carb_value_grams_interpolations', 0):,}")
        logger.info(f"   Large Gaps Skipped: {interp_analysis.get('large_gaps_skipped', 0):,}")
        logger.info(f"   Sequences Processed: {interp_analysis.get('sequences_processed', 0):,}")
    
    # Calibration Removal Analysis
    if 'calibration_removal_analysis' in stats and stats['calibration_removal_analysis']:
        removal_analysis = stats['calibration_removal_analysis']
        logger.info(f"\nCALIBRATION REMOVAL ANALYSIS:")
        logger.info(f"   Calibration Events Removed: {removal_analysis.get('calibration_events_removed', 0):,}")
        logger.info(f"   Records Before Removal: {removal_analysis.get('records_before_removal', 0):,}")
        logger.info(f"   Records After Removal: {removal_analysis.get('records_after_removal', 0):,}")
        logger.info(f"   Removal Enabled: {removal_analysis.get('calibration_removal_enabled', False)}")
    
    # Filtering Analysis
    if 'filtering_analysis' in stats and stats['filtering_analysis']:
        filter_analysis = stats['filtering_analysis']
        logger.info(f"\nSEQUENCE FILTERING ANALYSIS:")
        logger.info(f"   Original Sequences: {filter_analysis.get('original_sequences', 0):,}")
        logger.info(f"   Sequences After Filtering: {filter_analysis.get('filtered_sequences', 0):,}")
        logger.info(f"   Sequences Removed: {filter_analysis.get('removed_sequences', 0):,}")
        logger.info(f"   Original Records: {filter_analysis.get('original_records', 0):,}")
        logger.info(f"   Records After Filtering: {filter_analysis.get('filtered_records', 0):,}")
        logger.info(f"   Records Removed: {filter_analysis.get('removed_records', 0):,}")
    
    # Fixed-Frequency Analysis
    if 'fixed_frequency_analysis' in stats and stats['fixed_frequency_analysis']:
        fixed_freq_analysis = stats['fixed_frequency_analysis']
        logger.info(f"\nFIXED-FREQUENCY ANALYSIS:")
        logger.info(f"   Sequences Processed: {fixed_freq_analysis.get('sequences_processed', 0):,}")
        logger.info(f"   Time Adjustments Made: {fixed_freq_analysis.get('time_adjustments', 0):,}")
        logger.info(f"   Glucose Interpolations: {fixed_freq_analysis.get('glucose_interpolations', 0):,}")
        logger.info(f"   Insulin Records Shifted: {fixed_freq_analysis.get('insulin_shifted_records', 0):,}")
        logger.info(f"   Carb Records Shifted: {fixed_freq_analysis.get('carb_shifted_records', 0):,}")
        logger.info(f"   Records Before: {fixed_freq_analysis.get('total_records_before', 0):,}")
        logger.info(f"   Records After: {fixed_freq_analysis.get('total_records_after', 0):,}")
        
        # Data Density Analysis and Change Explanation
        if 'data_density_before' in fixed_freq_analysis and 'data_density_after' in fixed_freq_analysis:
            before_density = fixed_freq_analysis['data_density_before']
            after_density = fixed_freq_analysis['data_density_after']
            interval_minutes = preprocessor.expected_interval_minutes
            
            logger.info(f"\n   DATA DENSITY ({interval_minutes}-minute intervals):")
            logger.info(f"      Before: {before_density.get('avg_points_per_interval', 0.0):.2f} points/interval")
            logger.info(f"      After: {after_density.get('avg_points_per_interval', 0.0):.2f} points/interval")
            
            if 'density_change_explanation' in fixed_freq_analysis:
                explanation = fixed_freq_analysis['density_change_explanation']
                density_change = explanation.get('density_change_pct', 0.0)
                logger.info(f"      Density Change: {density_change:+.1f}%")
                logger.info(f"      Change Explained by Density: {explanation.get('explained_pct', 0.0):.1f}%")
    
    # Glucose Filtering Analysis
    if 'glucose_filtering_analysis' in stats and stats['glucose_filtering_analysis']:
        glucose_filter_analysis = stats['glucose_filtering_analysis']
        logger.info(f"\nGLUCOSE-ONLY FILTERING ANALYSIS:")
        logger.info(f"   Glucose-Only Mode Enabled: {glucose_filter_analysis.get('glucose_only_enabled', False)}")
        logger.info(f"   Original Records: {glucose_filter_analysis.get('original_records', 0):,}")
        logger.info(f"   Records After Filtering: {glucose_filter_analysis.get('records_after_filtering', 0):,}")
        logger.info(f"   Records Removed (No Glucose): {glucose_filter_analysis.get('records_removed', 0):,}")
        if glucose_filter_analysis.get('fields_removed', []):
            logger.info(f"   Fields Removed: {', '.join(glucose_filter_analysis['fields_removed'])}")
    
    # Data Quality
    quality = stats['data_quality']
    logger.info(f"\nDATA QUALITY:")
    logger.info(f"   Glucose Data Completeness: {quality.get('glucose_data_completeness', 0):.1f}%")
    
    # Handle both insulin completeness formats
    if 'insulin_data_completeness' in quality:
        logger.info(f"   Insulin Data Completeness: {quality['insulin_data_completeness']:.1f}%")
    else:
        fast_acting = quality.get('fast_acting_insulin_data_completeness', 0)
        long_acting = quality.get('long_acting_insulin_data_completeness', 0)
        if fast_acting > 0 or long_acting > 0:
            logger.info(f"   Fast-Acting Insulin Data Completeness: {fast_acting:.1f}%")
            logger.info(f"   Long-Acting Insulin Data Completeness: {long_acting:.1f}%")
        else:
            logger.info(f"   Insulin Data Completeness: 0.0%")
    
    logger.info(f"   Carb Data Completeness: {quality.get('carb_data_completeness', 0):.1f}%")
    logger.info(f"   Interpolated Records: {quality.get('interpolated_records', 0):,}")
    
    logger.info("\n" + "="*60)

