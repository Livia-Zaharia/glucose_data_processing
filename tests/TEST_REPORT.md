# Glucose ML Preprocessor Test Report

This report documents the test suite for `glucose_ml_preprocessor.py`, focusing on verifying each processing step and handling edge cases.

## Test Suite: `tests/test_glucose_ml_preprocessor_steps.py`

The test suite uses `pytest` and `polars` to verify the logic of `GlucoseMLPreprocessor` independently of external data files (using mocked data).

### 1. Initialization (`test_init_defaults`)

**What it tests:** Verifies that the preprocessor initializes with correct default configuration values.
**Input:** `GlucoseMLPreprocessor()`
**Expected Output:** Attributes like `expected_interval_minutes` match the default values (5).

### 2. Timestamp Parsing (`test_parse_timestamp`)

**What it tests:** Verifies the robustness of the timestamp parser against different formats and invalid inputs.
**Input:**

- Valid: `"2023-01-01 12:00:00"`, `"2023-01-01T12:00:00"`
- Invalid: `""`, `None`, `"invalid-date"`
  **Expected Output:**
- Valid: `datetime` object
- Invalid: `None`

### 3. Data Consolidation (`test_consolidate_glucose_data_dexcom_format`)

**What it tests:** Verifies that the preprocessor can discover and load CSV files from a directory (mocked as Dexcom format).
**Input:** Temporary directory containing a generated `dexcom_data.csv` with headers `Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL)`.
**Expected Output:** A Polars DataFrame with loaded data and correct columns.

### 4. Gap Detection (`test_detect_gaps_and_sequences_*`)

**What it tests:** Logic for identifying gaps in time series and assigning sequence IDs.

- **Continuous Data:**
  - **Input:** 20 records with 5-minute intervals.
  - **Expected Output:** 1 unique sequence ID, 0 gaps.
- **With Large Gap:**
  - **Input:** 5 records, 30-minute gap, 5 records.
  - **Expected Output:** 2 unique sequence IDs (split at the gap).
- **Multi-User:**
  - **Input:** 5 records for User 1, 5 records for User 2.
  - **Expected Output:** 2 unique sequence IDs, correctly offset/namespaced by user ID.

### 5. Interpolation (`test_interpolate_missing_values_small_gap`)

**What it tests:** Logic for filling small gaps (<= max gap size) with linear interpolation.
**Input:** Sequence with points at 10:00 and 10:10 (10-min gap). Config `expected_interval=5`.
**Expected Output:** 3 records (10:00, 10:05, 10:10). 10:05 is interpolated linearly.

### 6. Sequence Filtering (`test_filter_sequences_by_length`)

**What it tests:** Removing sequences shorter than `min_sequence_len`.
**Input:**

- Sequence 0: 3 records
- Sequence 1: 10 records
- Config `min_sequence_len=5`
  **Expected Output:** Only Sequence 1 remains. Stats show 1 sequence removed.

### 7. Fixed Frequency Creation (`test_create_fixed_frequency_data`)

**What it tests:** Aligning irregular timestamps to a fixed grid (e.g., every 5 minutes on the minute).
**Input:** Timestamps `10:00:30`, `10:05:45`.
**Expected Output:** Timestamps aligned to `10:01:00`, `10:06:00` (depending on start alignment logic) with 5-minute spacing. Glucose values interpolated to these new times.

### 8. Glucose Only Filtering (`test_filter_glucose_only`)

**What it tests:** Removing non-essential columns when `glucose_only=True`.
**Input:** DataFrame with Glucose, Carbs, Event Type.
**Expected Output:** DataFrame with only Glucose and timestamp/sequence columns.

### 9. ML Data Preparation (`test_prepare_ml_data`)

**What it tests:** Final formatting for ML training (column order, data types).
**Input:** DataFrame with mixed types (Glucose as string).
**Expected Output:** `sequence_id` as first column, numeric columns cast to Float64.

### 10. Full Integration (`test_process_integration`)

**What it tests:** The entire pipeline `process()` method using temporary files.
**Input:** 20 records of continuous synthetic data in a temp file.
**Expected Output:** A processed DataFrame with correct length, sequence IDs, and matching statistics.

## How to Run

```bash
pytest tests/test_glucose_ml_preprocessor_steps.py -v
```
