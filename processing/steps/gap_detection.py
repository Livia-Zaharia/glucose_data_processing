"""Gap detection and sequence creation logic."""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from processing.core.fields import StandardFieldNames

class GapDetector:
    """
    Detects time gaps and creates sequence IDs, marking calibration periods 
    and sequences for removal.
    """
    
    def __init__(
        self,
        small_gap_max_minutes: int,
        calibration_period_minutes: int,
        remove_after_calibration_hours: int
    ) -> None:
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        self.calibration_period_seconds = calibration_period_minutes * 60
        self.remove_after_calibration_hours = remove_after_calibration_hours
        
    def detect_gaps_and_sequences(
        self, 
        df: pl.DataFrame, 
        last_sequence_id: int = 0, 
        field_categories_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
        """
        Main entry point for gap detection and sequence creation.
        """
        logger.info("Detecting gaps and creating sequences...")
        
        ts_col = StandardFieldNames.TIMESTAMP
        user_id_col = StandardFieldNames.USER_ID
        seq_id_col = StandardFieldNames.SEQUENCE_ID

        calibration_stats = {
            'calibration_periods_detected': 0,
            'sequences_marked_for_removal': 0,
            'total_records_marked_for_removal': 0
        }
        
        current_last_sequence_id = last_sequence_id
        
        if user_id_col in df.columns:
            logger.info("Processing multi-user data - creating sequences per user...")
            all_sequences: List[pl.DataFrame] = []
            
            for user_id in sorted(df[user_id_col].unique().to_list()):
                user_data = df.filter(pl.col(user_id_col) == user_id).sort(ts_col)
                user_sequences, user_calib_stats, current_last_sequence_id = self._create_sequences_for_user(
                    user_data, current_last_sequence_id, user_id, field_categories_dict
                )
                all_sequences.append(user_sequences)
                
                calibration_stats['calibration_periods_detected'] += int(user_calib_stats['calibration_periods_detected'])
                calibration_stats['total_records_marked_for_removal'] += int(user_calib_stats['total_records_marked_for_removal'])
            
            if all_sequences:
                df = pl.concat(all_sequences).sort([user_id_col, seq_id_col, ts_col])
            else:
                df = df.clear()
        else:
            df = df.sort(ts_col)
            df, calibration_stats, current_last_sequence_id = self._create_sequences_for_user(
                df, current_last_sequence_id, None, field_categories_dict
            )
        
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
            stats = {
                'total_sequences': 0,
                'gap_positions': 0,
                'total_gaps': 0,
                'sequence_lengths': {},
                'calibration_period_analysis': calibration_stats
            }
        
        logger.info(f"Created {stats['total_sequences']} sequences")
        logger.info(f"Found {stats['total_gaps']} gaps > {self.small_gap_max_seconds / 60} minutes")
        
        if calibration_stats['calibration_periods_detected'] > 0:
            logger.info(f"Detected {calibration_stats['calibration_periods_detected']} calibration periods")
            logger.info(f"Removed {calibration_stats['total_records_marked_for_removal']} records after calibration")
        
        columns_to_remove = ['time_diff_seconds', 'is_gap', 'is_calibration_gap', 'remove_due_to_calibration']
        df = df.drop([col for col in columns_to_remove if col in df.columns])
        
        return df, stats, current_last_sequence_id

    def _create_sequences_for_user(
        self, 
        user_df: pl.DataFrame, 
        last_sequence_id: int = 0, 
        user_id: Optional[str] = None, 
        field_categories_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
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
        
        # Identify calibration gaps
        df = df.with_columns([
            (pl.col('time_diff_seconds') > self.calibration_period_seconds).fill_null(False).alias('is_calibration_gap')
        ])
        
        # Identify standard gaps
        df = df.with_columns([
            (pl.col('time_diff_seconds') > self.small_gap_max_seconds).fill_null(False).alias('is_gap')
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
                    
                    safe_field_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                    gap_columns.append(field_gap.alias(f'is_gap_{safe_field_name}'))
                
                if gap_columns:
                    df = df.with_columns(gap_columns)
                    gap_exprs = [pl.col('is_gap')]
                    for field in continuous_fields_to_check:
                        safe_field_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                        gap_exprs.append(pl.col(f'is_gap_{safe_field_name}'))
                    
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
                        
                        safe_field_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                        gap_columns.append(field_gap.alias(f'is_gap_{safe_field_name}'))
                    
                    if gap_columns:
                        df = df.with_columns(gap_columns)
                        gap_exprs = [pl.col('is_gap')]
                        for field in continuous_fields_to_check:
                            safe_field_name = field.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                            gap_exprs.append(pl.col(f'is_gap_{safe_field_name}'))
                        
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

