"""Interpolation logic for filling small gaps."""

import polars as pl
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from processing.core.fields import StandardFieldNames, INTERPOLATED_EVENT_TYPE

class ValueInterpolator:
    """
    Interpolates missing values for small gaps within sequences.
    """
    
    def __init__(self, expected_interval_minutes: int, small_gap_max_minutes: int) -> None:
        self.expected_interval_minutes = expected_interval_minutes
        self.small_gap_max_minutes = small_gap_max_minutes
        self.small_gap_max_seconds = small_gap_max_minutes * 60

    def interpolate_missing_values(
        self, 
        df: pl.DataFrame, 
        field_categories_dict: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Main entry point for interpolation.
        """
        ts_col = StandardFieldNames.TIMESTAMP
        glucose_col = StandardFieldNames.GLUCOSE_VALUE
        seq_id_col = StandardFieldNames.SEQUENCE_ID
        event_type_col = StandardFieldNames.EVENT_TYPE
        user_id_col = StandardFieldNames.USER_ID

        if field_categories_dict is None:
            field_categories_dict = {
                'continuous': [glucose_col],
                'occasional': [],
                'service': []
            }
        
        continuous_fields = field_categories_dict.get('continuous', [])
        if glucose_col in df.columns and glucose_col not in continuous_fields:
            continuous_fields.append(glucose_col)
        
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
        
        for field in fields_to_interpolate:
            interpolation_stats[f'{field_stats_keys[field]}_interpolations'] = 0
        
        interpolation_stats['sequences_processed'] = df[seq_id_col].n_unique()
        per_field_interpolations = {field: 0 for field in fields_to_interpolate}
        
        # Step 1: For each continuous field, fill missing values at existing timestamps
        for field in fields_to_interpolate:
            sequences = df[seq_id_col].unique().to_list()
            
            for seq_id in sequences:
                seq_mask = pl.col(seq_id_col) == seq_id
                seq_df = df.filter(seq_mask).sort(ts_col)
                
                non_null_rows = seq_df.filter(pl.col(field).is_not_null())
                
                if len(non_null_rows) < 2:
                    continue
                
                non_null_with_diff = non_null_rows.with_columns([
                    (pl.col(ts_col).diff().dt.total_seconds() / 60.0)
                    .alias('time_diff_minutes')
                ])
                
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
                    
                    missing_points = int((time_diff / self.expected_interval_minutes) - 1)
                    if missing_points <= 0:
                        continue
                    
                    for j in range(1, missing_points + 1):
                        interp_timestamp = prev_timestamp + timedelta(minutes=j * self.expected_interval_minutes)
                        
                        existing_rows = seq_df.filter(pl.col(ts_col) == interp_timestamp)
                        if existing_rows.height > 0 and existing_rows[field][0] is None:
                            alpha = (j * self.expected_interval_minutes) / time_diff
                            interp_value = float(prev_value + alpha * (curr_value - prev_value))
                            updates.append((interp_timestamp, interp_value))
                
                if updates:
                    per_field_interpolations[field] += len(updates)
                    
                    update_expr = pl.col(field)
                    for ts, val in updates:
                        update_expr = pl.when(
                            (pl.col(seq_id_col) == seq_id) & (pl.col(ts_col) == ts) & (pl.col(field).is_null())
                        ).then(pl.lit(val)).otherwise(update_expr)
                    
                    df = df.with_columns([update_expr.alias(field)])
                    
                    if event_type_col in df.columns:
                        event_update_expr = pl.col(event_type_col)
                        for ts, _ in updates:
                            event_update_expr = pl.when(
                                (pl.col(seq_id_col) == seq_id) & (pl.col(ts_col) == ts) & (pl.col(event_type_col) != INTERPOLATED_EVENT_TYPE)
                            ).then(pl.lit(INTERPOLATED_EVENT_TYPE)).otherwise(event_update_expr)
                        df = df.with_columns([event_update_expr.alias(event_type_col)])
        
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

