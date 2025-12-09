#!/usr/bin/env python3
"""
Tests for interpolate_missing_values() with continuous field categories.

Tests various scenarios:
1. Multiple continuous fields (some with same data points as glucose, some with fewer)
2. Continuous, occasional, and service fields together
3. Edge cases (missing values, null handling, etc.)
"""

import polars as pl
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from glucose_ml_preprocessor import GlucoseMLPreprocessor


def create_test_dataframe(
    timestamps: list,
    glucose_values: list,
    continuous_field1_values: list = None,
    continuous_field2_values: list = None,
    occasional_field_values: list = None,
    service_field_values: list = None
) -> pl.DataFrame:
    """Create a test DataFrame with specified values."""
    data = {
        'timestamp': timestamps,
        'Glucose Value (mg/dL)': glucose_values,
        'sequence_id': [1] * len(timestamps),
        'Timestamp (YYYY-MM-DDThh:mm:ss)': [ts.strftime('%Y-%m-%dT%H:%M:%S') for ts in timestamps],
        'Event Type': ['EGV'] * len(timestamps)
    }
    
    if continuous_field1_values is not None:
        data['Continuous Field 1'] = continuous_field1_values
    if continuous_field2_values is not None:
        data['Continuous Field 2'] = continuous_field2_values
    if occasional_field_values is not None:
        data['Occasional Field'] = occasional_field_values
    if service_field_values is not None:
        data['Service Field'] = service_field_values
    
    df = pl.DataFrame(data)
    # Timestamps are already datetime objects, no conversion needed
    return df


def test_multiple_continuous_fields_same_points():
    """Test interpolation with multiple continuous fields that have values at same points as glucose."""
    preprocessor = GlucoseMLPreprocessor(expected_interval_minutes=5, small_gap_max_minutes=15)
    
    # Create data with a 10-minute gap (should interpolate 1 point)
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=5),
        base_time + timedelta(minutes=15),  # 10-minute gap from previous
        base_time + timedelta(minutes=20),
    ]
    
    glucose = [100.0, 110.0, 130.0, 140.0]
    continuous1 = [50.0, 55.0, 65.0, 70.0]  # Same pattern as glucose
    continuous2 = [200.0, None, 260.0, 280.0]  # Missing value at second point
    
    df = create_test_dataframe(
        timestamps=timestamps,
        glucose_values=glucose,
        continuous_field1_values=continuous1,
        continuous_field2_values=continuous2,
        occasional_field_values=[None, None, 10.0, None],  # Occasional field
        service_field_values=['A', 'A', 'A', 'A']  # Service field
    )
    
    field_categories = {
        'continuous': ['Glucose Value (mg/dL)', 'Continuous Field 1', 'Continuous Field 2'],
        'occasional': ['Occasional Field'],
        'service': ['Service Field', 'Event Type']
    }
    
    result, stats = preprocessor.interpolate_missing_values(df, field_categories)
    
    # Should have 5 rows (4 original + 1 interpolated)
    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
    
    # Check interpolated point (at 10:10)
    interpolated_row = result.filter(
        pl.col('timestamp') == base_time + timedelta(minutes=10)
    )
    assert len(interpolated_row) == 1, "Should have one interpolated row"
    
    # Glucose should be interpolated: 110 + 0.5 * (130 - 110) = 120
    interpolated_glucose = interpolated_row['Glucose Value (mg/dL)'][0]
    assert abs(interpolated_glucose - 120.0) < 0.01, f"Expected glucose ~120, got {interpolated_glucose}"
    
    # Continuous Field 1 should be interpolated: 55 + 0.5 * (65 - 55) = 60
    interpolated_c1 = interpolated_row['Continuous Field 1'][0]
    assert abs(interpolated_c1 - 60.0) < 0.01, f"Expected Continuous Field 1 ~60, got {interpolated_c1}"
    
    # Continuous Field 2: prev is None, so interpolation should be None
    interpolated_c2 = interpolated_row['Continuous Field 2'][0]
    assert interpolated_c2 is None, f"Expected Continuous Field 2 to be None (prev was None), got {interpolated_c2}"
    
    # Occasional field should be None (not interpolated)
    interpolated_occ = interpolated_row['Occasional Field'][0]
    assert interpolated_occ is None, f"Expected Occasional Field to be None, got {interpolated_occ}"
    
    # Service field should be empty string (not interpolated)
    interpolated_service = interpolated_row['Service Field'][0]
    assert interpolated_service == '', f"Expected Service Field to be empty, got {interpolated_service}"
    
    # Event Type should be 'Interpolated'
    assert interpolated_row['Event Type'][0] == 'Interpolated', "Event Type should be 'Interpolated'"
    
    print("✓ test_multiple_continuous_fields_same_points passed")


def test_continuous_field_with_fewer_points():
    """Test interpolation when continuous field has fewer data points than glucose."""
    preprocessor = GlucoseMLPreprocessor(expected_interval_minutes=5, small_gap_max_minutes=15)
    
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=5),
        base_time + timedelta(minutes=15),  # 10-minute gap
        base_time + timedelta(minutes=20),
    ]
    
    glucose = [100.0, 110.0, 130.0, 140.0]
    # Continuous field has value at first point, None at second, then value after gap
    # This tests that we use the actual prev value (from row before gap), not skipping nulls
    continuous_sparse = [50.0, 55.0, 65.0, 70.0]  # Has values, but let's test with one missing
    
    df = create_test_dataframe(
        timestamps=timestamps,
        glucose_values=glucose,
        continuous_field1_values=continuous_sparse,
        occasional_field_values=[None, None, 10.0, None]
    )
    
    field_categories = {
        'continuous': ['Glucose Value (mg/dL)', 'Continuous Field 1'],
        'occasional': ['Occasional Field'],
        'service': ['Event Type']
    }
    
    result, stats = preprocessor.interpolate_missing_values(df, field_categories)
    
    # Check interpolated point
    interpolated_row = result.filter(
        pl.col('timestamp') == base_time + timedelta(minutes=10)
    )
    
    # Continuous Field 1: prev=55 (from 10:05), curr=65 (from 10:15), alpha=0.5
    # So interpolated = 55 + 0.5*(65-55) = 60
    interpolated_c1 = interpolated_row['Continuous Field 1'][0]
    assert interpolated_c1 is not None, "Continuous Field 1 should be interpolated"
    assert abs(interpolated_c1 - 60.0) < 0.01, f"Expected Continuous Field 1 ~60, got {interpolated_c1}"
    
    # Test case where prev is None - should result in None
    timestamps2 = [
        base_time,
        base_time + timedelta(minutes=5),
        base_time + timedelta(minutes=15),  # 10-minute gap
        base_time + timedelta(minutes=20),
    ]
    continuous_with_none = [50.0, None, 65.0, 70.0]  # None at second point
    
    df2 = create_test_dataframe(
        timestamps=timestamps2,
        glucose_values=glucose,
        continuous_field1_values=continuous_with_none
    )
    
    result2, _ = preprocessor.interpolate_missing_values(df2, field_categories)
    interpolated_row2 = result2.filter(
        pl.col('timestamp') == base_time + timedelta(minutes=10)
    )
    
    # When prev is None, interpolation should be None
    interpolated_c1_none = interpolated_row2['Continuous Field 1'][0]
    assert interpolated_c1_none is None, f"Expected None when prev is None, got {interpolated_c1_none}"
    
    print("✓ test_continuous_field_with_fewer_points passed")


def test_no_continuous_fields_except_glucose():
    """Test that glucose is always interpolated even if not in field_categories."""
    preprocessor = GlucoseMLPreprocessor(expected_interval_minutes=5, small_gap_max_minutes=15)
    
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=10),  # 10-minute gap (creates 1 interpolated point)
    ]
    
    glucose = [100.0, 130.0]
    
    df = create_test_dataframe(
        timestamps=timestamps,
        glucose_values=glucose,
        occasional_field_values=[None, 10.0]
    )
    
    # field_categories doesn't include glucose explicitly
    field_categories = {
        'continuous': [],
        'occasional': ['Occasional Field'],
        'service': ['Event Type']
    }
    
    result, stats = preprocessor.interpolate_missing_values(df, field_categories)
    
    # Should still interpolate glucose (always included)
    # 10-minute gap = 1 missing point, so 2 original + 1 interpolated = 3 total
    assert len(result) == 3, f"Expected 3 rows (2 original + 1 interpolated), got {len(result)}"
    
    interpolated_row = result.filter(
        pl.col('timestamp') == base_time + timedelta(minutes=5)
    )
    assert len(interpolated_row) == 1, "Should have interpolated point"
    
    # Glucose should be interpolated: 100 + 0.5 * (130 - 100) = 115
    interpolated_glucose = interpolated_row['Glucose Value (mg/dL)'][0]
    expected = 100.0 + 0.5 * (130.0 - 100.0)  # alpha = 5/10 = 0.5
    assert abs(interpolated_glucose - expected) < 0.01, f"Expected glucose ~{expected}, got {interpolated_glucose}"
    
    print("✓ test_no_continuous_fields_except_glucose passed")


def test_large_gap_not_interpolated():
    """Test that large gaps (> small_gap_max_minutes) are not interpolated."""
    preprocessor = GlucoseMLPreprocessor(expected_interval_minutes=5, small_gap_max_minutes=15)
    
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(minutes=20),  # 20-minute gap (too large)
    ]
    
    glucose = [100.0, 130.0]
    continuous1 = [50.0, 65.0]
    
    df = create_test_dataframe(
        timestamps=timestamps,
        glucose_values=glucose,
        continuous_field1_values=continuous1
    )
    
    field_categories = {
        'continuous': ['Glucose Value (mg/dL)', 'Continuous Field 1'],
        'occasional': [],
        'service': ['Event Type']
    }
    
    result, stats = preprocessor.interpolate_missing_values(df, field_categories)
    
    # Should not interpolate (gap too large)
    assert len(result) == 2, f"Expected 2 rows (no interpolation), got {len(result)}"
    assert stats['small_gaps_filled'] == 0, "Should not fill large gaps"
    assert stats['large_gaps_skipped'] == 1, "Should skip 1 large gap"
    
    print("✓ test_large_gap_not_interpolated passed")


def test_extract_field_categories():
    """Test the extract_field_categories method."""
    # Test with UoM database
    categories = GlucoseMLPreprocessor.extract_field_categories('uom')
    
    assert 'continuous' in categories
    assert 'occasional' in categories
    assert 'service' in categories
    
    # Glucose should be in continuous
    assert 'Glucose Value (mg/dL)' in categories['continuous'], "Glucose should be in continuous category"
    
    # Test with unknown database (should return default)
    categories_unknown = GlucoseMLPreprocessor.extract_field_categories('unknown')
    assert 'Glucose Value (mg/dL)' in categories_unknown['continuous'], "Should default to glucose only"
    
    print("✓ test_extract_field_categories passed")


if __name__ == '__main__':
    print("Running interpolation tests for continuous fields...\n")
    
    test_extract_field_categories()
    test_multiple_continuous_fields_same_points()
    test_continuous_field_with_fewer_points()
    test_no_continuous_fields_except_glucose()
    test_large_gap_not_interpolated()
    
    print("\n✅ All tests passed!")

