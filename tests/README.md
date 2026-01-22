g# Testing in Glucose Data Processing

This folder contains the test suite for the Glucose Data Processing pipeline. These tests ensure that the data conversion, cleaning, and preprocessing steps work correctly across different source formats.

## How to Run Tests

The tests are designed to be run using `pytest`. The easiest way to run them is via `uv`:

```bash
uv run pytest
```

### Running Specific Tests

To run a specific test file, use the command:
```bash
uv run pytest tests/<test_filename>.py
```

## Test Categories

Our tests are organized into three main levels based on what part of the data flow they validate:

### I. Raw Data (Input Validation & Conversion)
Tests that run on original, unprocessed datasets to verify their integrity and our ability to read them correctly.

- **`test_database_converters_real_data.py`**
  ```bash
  uv run pytest tests/test_database_converters_real_data.py
  ```
  **What it checks:** Verifies that our system correctly detects different database formats (HUPA, UoM, Medtronic, etc.) and can consolidate multi-file datasets into a single dataframe without losing information.

- **`test_raw_data_quality.py`**
  ```bash
  uv run pytest tests/test_raw_data_quality.py
  ```
  **What it checks:** Scans raw datasets for unexpected large time gaps (e.g., more than several days) per user. This is a pre-processing quality check to avoid using fragmented data.

- **`test_download.py`**
  ```bash
  uv run pytest tests/test_download.py
  ```
  **What it checks:** Validates the programmatic downloading logic for public datasets from Zenodo, Figshare, Mendeley, and PhysioNet.

---

### II. Processed Data (Output Validation & Consistency)
Tests that run on the final results of the processing pipeline or compare outputs across different versions to ensure consistency.

- **`test_processed_data_quality.py`**
  ```bash
  uv run pytest tests/test_processed_data_quality.py
  ```
  **What it checks:** Specifically ensures that processing large gaps doesn't cause "interpolation explosion" (creating thousands of fake rows).

- **`test_compare_checkpoints.py`**
  ```bash
  uv run pytest tests/test_compare_checkpoints.py
  ```
  **What it checks:** Performs regression testing by comparing current processing results against known-good "checkpoints" stored in Parquet format.

- **`test_output_fields_new_field_addition.py`**
  ```bash
  uv run pytest tests/test_output_fields_new_field_addition.py
  ```
  **What it checks:** Verifies that new fields added to the configuration or schemas appear correctly in the final output files without breaking existing logic.

---

### III. Pipeline & Logic (Internal Methods & Workflows)
Tests that validate individual processing steps, mathematical methods, and the end-to-end processing workflow using controlled mock data.

- **`test_glucose_ml_preprocessor_steps.py`**
  ```bash
  uv run pytest tests/test_glucose_ml_preprocessor_steps.py
  ```
  **What it checks:** Tests the full end-to-end execution of the ML preprocessor, ensuring all steps (gap detection, interpolation, filtering, casting) happen in the correct sequence.

- **`test_fixed_frequency_logic.py`**
  ```bash
  uv run pytest tests/test_fixed_frequency_logic.py
  ```
  **What it checks:** Validates the core logic for resampling data to a fixed interval (e.g., 5 mins), including how it shifts events to the nearest timestamp and sums simultaneous occurrences.

- **`test_fixed_frequency_continuous_fields.py`**
  ```bash
  uv run pytest tests/test_fixed_frequency_continuous_fields.py
  ```
  **What it checks:** Specifically tests the resampling and fixed-grid alignment for continuous fields like glucose readings.

- **`test_fixed_frequency_edge_cases.py`**
  ```bash
  uv run pytest tests/test_fixed_frequency_edge_cases.py
  ```
  **What it checks:** Handles edge cases for resampling, such as empty inputs, datasets with only one row, or extreme time jitter.

- **`test_interpolate_continuous_fields.py`**
  ```bash
  uv run pytest tests/test_interpolate_continuous_fields.py
  ```
  **What it checks:** Validates the linear interpolation logic used to fill small gaps (typically <= 15 minutes) within data sequences.

- **`test_loop_deduplication.py`**
  ```bash
  uv run pytest tests/test_loop_deduplication.py
  ```
  **What it checks:** Specifically tests the removal of duplicate records, which are common in data exported from Loop database systems.

- **`test_calibration_schema_logic.py`**
  ```bash
  uv run pytest tests/test_calibration_schema_logic.py
  ```
  **What it checks:** Verifies that our internal schemas correctly map source-specific field names to our standardized internal format.

---

## Testing Principles

- **Real Data + Ground Truth:** We prefer using actual source data (found in the `DATA/` folder) and computing expected values at runtime.
- **Deterministic Coverage:** We use fixed seeds and explicit filters to include both representative and edge cases.
- **Avoid "Happy Path" Only:** We actively test for nulls, empty strings, and malformed data to ensure the pipeline is robust.

## Prerequisites for Testing

Some tests require the `DATA/` folder to be populated. You can download public datasets using:

```bash
# List available datasets
uv run python download.py list

# Download a specific dataset (e.g., T1D-UOM)
uv run python download.py by-name T1D-UOM
```

## Adding New Tests

1. Add a representative test case to the relevant file based on the categories above.
2. Ensure you provide the execution command in the same format as existing tests.
3. Ensure all tests pass (`uv run pytest`) before submitting changes.
