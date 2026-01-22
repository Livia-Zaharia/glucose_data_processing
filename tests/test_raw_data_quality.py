import pytest
import polars as pl
from datetime import datetime, timedelta
from processing.core.fields import StandardFieldNames

class TestRawDataQuality:
    """
    Tests for validating the quality and integrity of raw datasets before processing.
    """

    def test_detect_large_gaps_in_raw_data(self):
        """
        Check that we can detect large gaps (several days) in raw data per user.
        This represents a 'raw data quality' check requested by the user.
        """
        ts_col = StandardFieldNames.TIMESTAMP
        user_col = StandardFieldNames.USER_ID
        
        # Mock data with a 5-day gap for one user
        start_ts = datetime(2023, 1, 1, 12, 0)
        gap_start = start_ts + timedelta(hours=1)
        gap_end = gap_start + timedelta(days=5)
        
        raw_df = pl.DataFrame({
            user_col: ["user1", "user1", "user1"],
            ts_col: [start_ts, gap_start, gap_end],
            StandardFieldNames.GLUCOSE_VALUE: [100.0, 105.0, 110.0]
        })
        
        # Logic to detect gaps > 3 days
        max_gap_allowed = timedelta(days=3).total_seconds()
        
        # Calculate gaps per user
        gaps_df = raw_df.sort([user_col, ts_col]).with_columns([
            pl.col(ts_col).diff().over(user_col).dt.total_seconds().alias("gap_seconds")
        ])
        
        large_gaps = gaps_df.filter(pl.col("gap_seconds") > max_gap_allowed)
        
        # We expect to find 1 large gap in our mock data
        assert len(large_gaps) == 1
        assert large_gaps[user_col][0] == "user1"

if __name__ == "__main__":
    pytest.main([__file__])
