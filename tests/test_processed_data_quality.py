import pytest
import polars as pl
from datetime import datetime, timedelta
from glucose_ml_preprocessor import GlucoseMLPreprocessor, _run_processing_pipeline
from processing.core.fields import StandardFieldNames

class TestProcessedDataQuality:
    """
    Tests for ensuring the processed output does not contain artifacts or errors 
    introduced by the pipeline logic.
    """

    def test_interpolation_explosion_prevention(self):
        """
        Verify that processing a dataset with large gaps doesn't result in 
        an 'interpolation explosion' (creating thousands of unnecessary rows).
        """
        ts_col = StandardFieldNames.TIMESTAMP
        user_col = StandardFieldNames.USER_ID
        
        # Setup preprocessor with a 15-min small gap limit
        # but provide a 2-hour gap in data.
        preprocessor = GlucoseMLPreprocessor(
            expected_interval_minutes=5,
            small_gap_max_minutes=15,
            create_fixed_frequency=True,
            min_sequence_len=1 # Ensure it doesn't filter out our small mock data
        )
        
        start_ts = datetime(2023, 1, 1, 12, 0)
        end_ts = start_ts + timedelta(hours=2) # 120 minutes gap
        
        df = pl.DataFrame({
            user_col: ["user1", "user1"],
            ts_col: [start_ts, end_ts],
            StandardFieldNames.GLUCOSE_VALUE: [100.0, 110.0],
            StandardFieldNames.EVENT_TYPE: ["EGV", "EGV"]
        })
        
        field_categories = {"continuous": [StandardFieldNames.GLUCOSE_VALUE], "occasional": [], "service": []}
        
        # Step 2 & 3 are where the magic (or explosion) happens
        processed_df, _, _ = _run_processing_pipeline(
            df, 
            last_sequence_id=0, 
            field_categories_dict=field_categories,
            gap_detector=preprocessor.gap_detector,
            interpolator=preprocessor.interpolator,
            filter_step=preprocessor.filter_step,
            fixed_freq_generator=preprocessor.fixed_freq_generator,
            ml_preparer=preprocessor.ml_preparer,
            stats_manager=preprocessor.stats_manager,
            create_fixed_frequency=True
        )
        
        # If it exploded, we would have 120/5 = 24 points.
        # But since the gap (120 min) > small_gap_max (15 min), 
        # it should NOT have interpolated. 
        
        # Check that we didn't get a massive amount of rows
        assert len(processed_df) <= 10, f"Interpolation explosion! Found {len(processed_df)} rows for a 2-hour gap."

if __name__ == "__main__":
    pytest.main([__file__])
