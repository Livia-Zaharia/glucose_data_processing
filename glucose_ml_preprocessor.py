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
from datetime import datetime
from pathlib import Path
from loguru import logger
import warnings
import yaml
import sys
import json

# Import database detection and conversion classes
from formats import DatabaseDetector
from formats.base_converter import CSVFormatConverter

# Import modular processing components
from processing.core.fields import StandardFieldNames, INTERPOLATED_EVENT_TYPE
from processing.core.config import extract_field_categories
from processing.steps.gap_detection import GapDetector
from processing.steps.interpolation import ValueInterpolator
from processing.steps.filtering import SequenceFilter
from processing.steps.fixed_frequency import FixedFreqGenerator
from processing.steps.ml_prep import MLDataPreparer
from processing.stats_manager import StatsManager, print_statistics as sm_print_statistics

warnings.filterwarnings('ignore')

# Constants for common field names and literal values
DEFAULT_STREAMING_MAX_BUFFER_MB = 256
DEFAULT_STREAMING_FLUSH_MAX_USERS = 10
MIN_BUFFER_MB = 32

class GlucoseMLPreprocessor:
    """
    Preprocessor for glucose monitoring data to prepare it for machine learning.
    Acts as a facade delegating work to specialized processing steps.
    """
    
    @classmethod
    def from_config_file(cls, config_path: Path, **cli_overrides: Any) -> "GlucoseMLPreprocessor":
        """
        Create a GlucoseMLPreprocessor instance from a YAML configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        CSVFormatConverter.initialize_from_config(config)
        
        dexcom_config = config.get('dexcom', {})
        high_value = dexcom_config.get('high_glucose_value', 401)
        low_value = dexcom_config.get('low_glucose_value', 39)
        
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
        
        self.fields = StandardFieldNames()
        self._original_record_count: int = 0
        self._field_categories_dict: Optional[Dict[str, Any]] = None
        
        # Initialize specialized processing steps
        self.gap_detector = GapDetector(
            small_gap_max_minutes=small_gap_max_minutes,
            calibration_period_minutes=calibration_period_minutes,
            remove_after_calibration_hours=remove_after_calibration_hours
        )
        self.interpolator = ValueInterpolator(
            expected_interval_minutes=expected_interval_minutes,
            small_gap_max_minutes=small_gap_max_minutes
        )
        self.filter_step = SequenceFilter(
            min_sequence_len=min_sequence_len,
            glucose_only=glucose_only
        )
        self.fixed_freq_generator = FixedFreqGenerator(
            expected_interval_minutes=expected_interval_minutes
        )
        self.ml_preparer = MLDataPreparer(config=self.config)
        self.stats_manager = StatsManager()

    @staticmethod
    def extract_field_categories(database_type: str) -> Dict[str, Any]:
        """Legacy static method for field categories extraction."""
        return extract_field_categories(database_type)

    def _df_estimated_size_bytes(self, df: pl.DataFrame) -> int:
        try:
            return int(df.estimated_size())
        except (AttributeError, ValueError):
            return int(len(df) * max(1, len(df.columns)) * 16)

    def _streaming_buffer_max_bytes(self) -> int:
        mb = self.config.get("streaming_max_buffer_mb", DEFAULT_STREAMING_MAX_BUFFER_MB)
        try:
            mb_i = int(mb)
        except (ValueError, TypeError):
            mb_i = DEFAULT_STREAMING_MAX_BUFFER_MB
        return max(MIN_BUFFER_MB, mb_i) * 1024 * 1024

    def _streaming_flush_max_users(self) -> int:
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
        field_to_display = CSVFormatConverter.get_field_to_display_name_map()
        seq_id_col = StandardFieldNames.SEQUENCE_ID

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

            df, _gap_stats, current_last_sequence_id = self.gap_detector.detect_gaps_and_sequences(
                user_df, current_last_sequence_id, field_categories_dict
            )
            df, _ = self.interpolator.interpolate_missing_values(df, field_categories_dict)
            
            # Update filter step with current facade state for tests
            self.filter_step.min_sequence_len = self.min_sequence_len
            self.filter_step.glucose_only = self.glucose_only
            
            df, _ = self.filter_step.filter_sequences_by_length(df)
            if self.create_fixed_frequency:
                df, _ = self.fixed_freq_generator.create_fixed_frequency_data(df, field_categories_dict)
            
            df, _ = self.filter_step.filter_glucose_only(df)
            ml_df = self.ml_preparer.prepare_ml_data(df, field_categories_dict)

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

    def consolidate_glucose_data(self, data_folder: Path, output_file: Optional[Path] = None) -> pl.DataFrame:
        db_detector = DatabaseDetector()
        database_type = db_detector.detect_database_type(data_folder)
        logger.info(f"Detected database type: {database_type}")
        
        if database_type == 'unknown':
            raise ValueError(f"Could not detect database type for folder: {data_folder}")
        
        database_converter = db_detector.get_database_converter(database_type, self.config or {})
        if database_converter is None:
            raise ValueError(f"No converter available for database type: {database_type}")
        
        logger.info(f"Using {database_converter.get_database_name()}")
        df = database_converter.consolidate_data(data_folder, output_file)
        self._original_record_count = len(df)
        self.stats_manager.original_record_count = len(df)
        return df

    def detect_gaps_and_sequences(self, df: pl.DataFrame, last_sequence_id: int = 0, field_categories_dict: Optional[Dict[str, Any]] = None) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        return self.gap_detector.detect_gaps_and_sequences(df, last_sequence_id, field_categories_dict)

    def interpolate_missing_values(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, List[str]]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        return self.interpolator.interpolate_missing_values(df, field_categories_dict)

    def filter_sequences_by_length(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        self.filter_step.min_sequence_len = self.min_sequence_len
        return self.filter_step.filter_sequences_by_length(df)

    def create_fixed_frequency_data(self, df: pl.DataFrame, field_categories_dict: Optional[Dict[str, List[str]]] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        return self.fixed_freq_generator.create_fixed_frequency_data(df, field_categories_dict)

    def filter_glucose_only(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        self.filter_step.glucose_only = self.glucose_only
        return self.filter_step.filter_glucose_only(df)

    def prepare_ml_data(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.ml_preparer.prepare_ml_data(df, self._field_categories_dict)

    def get_statistics(self, df: pl.DataFrame, gap_stats: Dict[str, Any], interp_stats: Dict[str, Any], removal_stats: Optional[Dict[str, Any]] = None, filter_stats: Optional[Dict[str, Any]] = None, replacement_stats: Optional[Dict[str, Any]] = None, glucose_filter_stats: Optional[Dict[str, Any]] = None, fixed_freq_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.stats_manager.get_statistics(df, gap_stats, interp_stats, filter_stats, glucose_filter_stats, fixed_freq_stats)

    def process(self, csv_folder: Path, output_file: Optional[Path] = None, last_sequence_id: int = 0) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        logger.info("Starting glucose data preprocessing for ML...")
        
        db_detector = DatabaseDetector()
        database_type = db_detector.detect_database_type(csv_folder)
        field_categories_dict = extract_field_categories(database_type) if database_type != 'unknown' else None
        self._field_categories_dict = field_categories_dict
        
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
        
        logger.info("STEP 1: Consolidating CSV files (mandatory step)...")
        df = self.consolidate_glucose_data(csv_folder)
        
        logger.info("STEP 2: Detecting gaps and creating sequences...")
        df, gap_stats, last_sequence_id = self.detect_gaps_and_sequences(df, last_sequence_id, field_categories_dict)
        
        logger.info("STEP 3: Interpolating missing values...")
        df, interp_stats = self.interpolate_missing_values(df, field_categories_dict)
        
        logger.info("STEP 4: Filtering sequences by minimum length...")
        df, filter_stats = self.filter_sequences_by_length(df)
        
        if self.create_fixed_frequency:
            logger.info("STEP 5: Creating fixed-frequency data...")
            df, fixed_freq_stats = self.create_fixed_frequency_data(df, field_categories_dict)
        else:
            fixed_freq_stats = {}
        
        logger.info("STEP 6: Filtering to glucose-only data...")
        df, glucose_filter_stats = self.filter_glucose_only(df)
        
        logger.info("STEP 7: Preparing final ML dataset...")
        ml_df = self.prepare_ml_data(df)
        
        stats = self.get_statistics(ml_df, gap_stats, interp_stats, filter_stats, glucose_filter_stats, fixed_freq_stats)
        
        if output_file:
            ml_df.write_csv(output_file)
            logger.info(f"Final processed data saved to: {output_file}")
        
        return ml_df, stats, last_sequence_id

    def process_multiple_databases(self, csv_folders: List[Path], output_file: Optional[Path] = None, last_sequence_id: int = 0) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        logger.info(f"Starting multi-database processing for {len(csv_folders)} databases...")
        
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
                if converter is not None and callable(getattr(converter, "iter_user_event_frames", None)):
                    field_categories_dict = extract_field_categories(db_type) if db_type != "unknown" else None
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
                        df, _gap_stats, current_last_sequence_id = self.gap_detector.detect_gaps_and_sequences(
                            user_df, current_last_sequence_id, field_categories_dict
                        )
                        df, _ = self.interpolator.interpolate_missing_values(df, field_categories_dict)
                        self.filter_step.min_sequence_len = self.min_sequence_len
                        df, _ = self.filter_step.filter_sequences_by_length(df)
                        if self.create_fixed_frequency:
                            df, _ = self.fixed_freq_generator.create_fixed_frequency_data(df, field_categories_dict)
                        self.filter_step.glucose_only = self.glucose_only
                        df, _ = self.filter_step.filter_glucose_only(df)
                        ml_df = self.ml_preparer.prepare_ml_data(df, field_categories_dict)
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
                    ml_df, stats, current_last_sequence_id = self.process(csv_folder, output_file=None, last_sequence_id=current_last_sequence_id)
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
                "sequence_analysis": {"sequence_lengths": {"count": total_sequences}, "longest_sequence": 0, "shortest_sequence": 0},
                "gap_analysis": {}, "interpolation_analysis": {}, "calibration_removal_analysis": {},
                "filtering_analysis": {}, "replacement_analysis": {}, "fixed_frequency_analysis": {},
                "glucose_filtering_analysis": {}, "data_quality": {},
            }
            placeholder = pl.DataFrame({seq_id_col: pl.Series([], dtype=pl.Int64)})
            return placeholder, aggregated_stats, current_last_sequence_id

        all_dataframes: List[pl.DataFrame] = []
        all_statistics: List[Dict[str, Any]] = []
        current_last_sequence_id = last_sequence_id
        
        for idx, csv_folder in enumerate(csv_folders, 1):
            ml_df, stats, current_last_sequence_id = self.process(csv_folder, output_file=None, last_sequence_id=current_last_sequence_id)
            if user_id_col in ml_df.columns:
                ml_df = ml_df.drop(user_id_col)
            
            max_seq_id = ml_df[seq_id_col].max() if len(ml_df) > 0 else current_last_sequence_id
            min_seq_id = ml_df[seq_id_col].min() if len(ml_df) > 0 else current_last_sequence_id
            stats['database_info'] = {
                'database_index': idx,
                'database_path': str(csv_folder),
                'sequence_id_range': {'min': int(min_seq_id) if min_seq_id is not None else current_last_sequence_id, 'max': int(max_seq_id) if max_seq_id is not None else current_last_sequence_id}
            }
            all_dataframes.append(ml_df)
            all_statistics.append(stats)
        
        combined_df = pl.concat(all_dataframes).sort([seq_id_col, ts_col])
        combined_stats = self.stats_manager.aggregate_statistics(all_statistics, csv_folders)
        
        if output_file:
            combined_df.write_csv(output_file)
        
        return combined_df, combined_stats, current_last_sequence_id

    # --- Legacy methods for backward compatibility with tests ---

    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Legacy method for timestamp parsing."""
        if not timestamp_str or timestamp_str.strip() == "":
            return None
        timestamp_str = timestamp_str.strip()
        formats = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        return None

    def _create_sequences_for_user(self, *args, **kwargs) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """Legacy private method delegation."""
        return self.gap_detector._create_sequences_for_user(*args, **kwargs)

    def _shift_events_rounding(self, *args, **kwargs) -> pl.DataFrame:
        """Legacy private method delegation."""
        return self.fixed_freq_generator._shift_events_rounding(*args, **kwargs)

def print_statistics(stats: Dict[str, Any], preprocessor: 'GlucoseMLPreprocessor' = None) -> None:
    params = None
    if preprocessor:
        params = {
            'expected_interval_minutes': preprocessor.expected_interval_minutes,
            'small_gap_max_minutes': preprocessor.small_gap_max_minutes,
            'remove_calibration': preprocessor.remove_calibration,
            'min_sequence_len': preprocessor.min_sequence_len,
            'calibration_period_minutes': preprocessor.calibration_period_minutes,
            'remove_after_calibration_hours': preprocessor.remove_after_calibration_hours,
            'create_fixed_frequency': preprocessor.create_fixed_frequency
        }
    sm_print_statistics(stats, params)
