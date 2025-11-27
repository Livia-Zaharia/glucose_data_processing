# Glucose Data Preprocessing for Machine Learning

A comprehensive tool for preprocessing glucose monitoring data from continuous glucose monitors (CGM) to prepare it for machine learning applications. This project handles data consolidation, gap detection, interpolation, calibration smoothing, and sequence filtering.

## 🚀 Quick Start

### 1. Setup Project Using UV

This project uses [UV](https://docs.astral.sh/uv/) as the package manager for fast and reliable dependency management.

#### Prerequisites

- Python 3.11 or higher
- UV package manager

#### Installation

1. **Install UV** (if not already installed):

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or using pip
   pip install uv
   ```

2. **Clone and setup the project**:

   ```bash
   git clone <your-repo-url>
   cd glucose_data_processing

   # Install dependencies using UV
   uv sync

   # Activate the virtual environment
   uv shell
   ```

3. **Verify installation**:
   ```bash
   python glucose_cli.py --help
   ```

### 2. Dependencies

The project requires the following Python packages (automatically managed by UV):

- **pandas** (≥2.3.2) - Data manipulation and analysis
- **polars** (≥1.33.1) - Fast DataFrame library for large datasets
- **pyarrow** (≥21.0.0) - Columnar data format for efficient data processing
- **typer** (≥0.12.0) - Command-line interface framework
- **pyyaml** (≥6.0) - YAML configuration file support

All dependencies are specified in `pyproject.toml` and will be automatically installed with `uv sync`.

## 📋 Project Overview

### What is this project about?

This project is designed to preprocess continuous glucose monitoring (CGM) data for machine learning applications. It addresses common challenges in glucose data:

- **Data Fragmentation**: CGM data often comes in multiple CSV files that need consolidation
- **Time Gaps**: Missing data points due to device disconnections or calibration periods
- **Data Quality Issues**: Spikes during calibration events and irregular sampling intervals
- **Sequence Management**: Identifying continuous data segments suitable for ML training

### Project Goals

1. **Data Consolidation**: Merge multiple CSV files into a single, chronologically ordered dataset
2. **Data Standardization**: Replace textual glucose values (High/Low) with numeric equivalents
3. **Calibration Removal**: Remove calibration events to create interpolatable gaps
4. **Gap Detection**: Identify time gaps and create sequence boundaries
5. **Smart Interpolation**: Fill small gaps while preserving sequence integrity
6. **Fixed-Frequency Creation**: Align sequences to consistent time intervals for ML models
7. **ML-Ready Output**: Generate clean, continuous sequences suitable for time-series ML models

## 🖥️ CLI Usage

### Basic Usage

**Single Database:**

```bash
python glucose_cli.py <input_folder> [OPTIONS]
```

**Multiple Databases (NEW!):**

```bash
python glucose_cli.py <input_folder1> <input_folder2> [<input_folder3> ...] [OPTIONS]
```

You can now process multiple databases with different formats into a single unified output file. The system automatically tracks sequence IDs across databases to ensure consistency.

### Command Line Options

| Option                                   | Short | Default                | Description                                             |
| ---------------------------------------- | ----- | ---------------------- | ------------------------------------------------------- |
| `--output`, `-o`                         |       | `glucose_ml_ready.csv` | Output file path for ML-ready data                      |
| `--interval`, `-i`                       |       | `5`                    | Time discretization interval in minutes                 |
| `--gap-max`, `-g`                        |       | `15`                   | Maximum gap size to interpolate in minutes              |
| `--min-length`, `-l`                     |       | `200`                  | Minimum sequence length to keep for ML training         |
| `--calibration/--no-calibration`         |       | `True`                 | Interpolate calibration glucose values                  |
| `--verbose`, `-v`                        |       | `False`                | Enable verbose output                                   |
| `--stats/--no-stats`                     |       | `True`                 | Show processing statistics                              |
| `--save-intermediate`, `-s`              |       | `False`                | Save intermediate files after each step                 |
| `--calibration-period`, `-c`             |       | `165`                  | Gap duration considered as calibration period (minutes) |
| `--remove-after-calibration`, `-r`       |       | `24`                   | Hours of data to remove after calibration period        |
| `--glucose-only`                         |       | `False`                | Output only glucose data with simplified fields         |
| `--fixed-frequency/--no-fixed-frequency` |       | `True`                 | Create fixed-frequency data with consistent intervals   |

### Examples

#### Single Database Processing

```bash
# Basic processing with default settings
python glucose_cli.py ./000-csv

# Custom parameters with verbose output
python glucose_cli.py ./000-csv --output my_glucose_data.csv --interval 5 --gap-max 10 --verbose

# Process with calibration period detection and data removal
python glucose_cli.py ./000-csv --calibration-period 165 --remove-after-calibration 24 --save-intermediate

# Quick processing without statistics
python glucose_cli.py ./000-csv --no-stats --output quick_output.csv

# Use configuration file
python glucose_cli.py ./000-csv --config glucose_config.yaml

# Override config file parameters with CLI arguments
python glucose_cli.py ./000-csv --config glucose_config.yaml --interval 10 --gap-max 30

# Process with glucose-only output (simplified format)
python glucose_cli.py ./000-csv --glucose-only --output glucose_only_data.csv

# Combine glucose-only with other options
python glucose_cli.py ./000-csv --glucose-only --min-length 100 --verbose

# Disable fixed-frequency data creation (use original irregular intervals)
python glucose_cli.py ./000-csv --no-fixed-frequency --output irregular_data.csv

# Enable fixed-frequency with custom interval
python glucose_cli.py ./000-csv --fixed-frequency --interval 10 --output fixed_10min_data.csv
```

#### Multi-Database Processing (NEW!)

```bash
# Combine multiple databases with different formats into one file
python glucose_cli.py ./000-csv ./libre3 ./zendo_small --output combined_ml_data.csv

# Process two Dexcom databases with verbose output
python glucose_cli.py ./000-csv ./000_small --output combined_dexcom.csv --verbose

# Combine different database types (Dexcom, Libre3, UoM)
python glucose_cli.py ./dexcom_data ./libre3_data ./uom_data --output all_formats_combined.csv

# Multi-database with custom parameters
python glucose_cli.py ./db1 ./db2 ./db3 --interval 5 --min-length 200 --glucose-only

# Use config file with multiple databases
python glucose_cli.py ./000-csv ./libre3 --config glucose_config.yaml --output combined.csv

# Combine databases with intermediate file saving
python glucose_cli.py ./db1 ./db2 --save-intermediate --verbose --output multi_db_output.csv
```

## ⚙️ YAML Configuration File

### Overview

For complex configurations or repeated processing, you can use a YAML configuration file to specify all parameters. Command line arguments always take precedence over configuration file values.

### Configuration File Usage

```bash
# Use configuration file
python glucose_cli.py ./000-csv --config glucose_config.yaml

# Override specific parameters from config file
python glucose_cli.py ./000-csv --config glucose_config.yaml --interval 10 --gap-max 30

# Use config file with custom output
python glucose_cli.py ./000-csv --config glucose_config.yaml --output custom_output.csv
```

### Configuration File Structure

The `glucose_config.yaml` file contains the essential configuration options:

```yaml
# Basic processing parameters
expected_interval_minutes: 5
small_gap_max_minutes: 15
remove_calibration: true
min_sequence_len: 200
save_intermediate_files: false

# Calibration period detection
calibration_period_minutes: 165
remove_after_calibration_hours: 24

# Glucose value replacement (configurable High/Low values)
glucose_value_replacement:
  high_value: 401 # Replace 'High' with 401 mg/dL
  low_value: 39 # Replace 'Low' with 39 mg/dL
  enabled: true

# Glucose-only mode (simplified output)
glucose_only: false # Output only glucose data with simplified fields

# Fixed-frequency data creation
create_fixed_frequency: true # Create fixed-frequency data with consistent intervals
```

### Key Configuration Features

#### **Core Processing Parameters**

- **`expected_interval_minutes`**: Time discretization interval (default: 5 minutes)
- **`small_gap_max_minutes`**: Maximum gap size to interpolate (default: 15 minutes)
- **`remove_calibration`**: Remove calibration events to create interpolatable gaps
- **`min_sequence_len`**: Minimum sequence length for ML training (default: 200 records)
- **`save_intermediate_files`**: Save intermediate files for debugging
- **`create_fixed_frequency`**: Create fixed-frequency data with consistent intervals (default: true)

#### **Calibration Period Detection**

- **`calibration_period_minutes`**: Gap duration considered as calibration period (default: 165 minutes)
- **`remove_after_calibration_hours`**: Hours of data to remove after calibration (default: 24 hours)

#### **Configurable High/Low Values**

- **`high_value`**: Numeric value to replace 'High' glucose readings (default: 401 mg/dL)
- **`low_value`**: Numeric value to replace 'Low' glucose readings (default: 39 mg/dL)
- **`enabled`**: Enable/disable High/Low value replacement

#### **Glucose-Only Mode**

- **`glucose_only`**: When enabled, outputs only glucose data with simplified fields (default: false)

#### **Fixed-Frequency Data Creation**

- **`create_fixed_frequency`**: When enabled, creates fixed-frequency data with consistent intervals (default: true)

### Priority Order

1. **Command line arguments** (highest priority)
2. **Configuration file values**
3. **Default values** (lowest priority)

Example:

```bash
# Config file sets interval=5, CLI overrides to 10
python glucose_cli.py ./000-csv --config glucose_config.yaml --interval 10
# Result: interval=10 (CLI wins)
```

### Creating Your Own Configuration

1. Copy `glucose_config.yaml` to create your custom configuration
2. Modify values as needed for your specific use case (focus on the core parameters)
3. Use the `--config` parameter to specify your file

```bash
# Use custom configuration
python glucose_cli.py ./000-csv --config my_custom_config.yaml
```

**Note**: The configuration file includes only the essential parameters that are actively used by the preprocessing pipeline. This keeps the configuration simple and focused on the most important settings.

## 🔀 Multi-Database Processing (NEW!)

### Overview

The system now supports processing multiple databases with different formats simultaneously, combining them into a single unified output file. This feature is particularly useful when you have glucose data from multiple sources (Dexcom, Libre3, UoM, etc.) and want to create a comprehensive dataset.

### Key Features

#### **1. Automatic Format Detection**

Each database folder is automatically detected and processed using the appropriate converter:

- Dexcom G6 format
- FreeStyle Libre 3 format
- University of Manchester T1D format
- Other supported formats

#### **2. Consistent Sequence ID Tracking**

The system ensures that sequence IDs remain unique and consistent across all databases:

- Database 1: Sequences 0-99
- Database 2: Sequences 100-199 (offset by 100)
- Database 3: Sequences 200-299 (offset by 200)
- And so on...

This prevents ID conflicts and maintains data integrity.

#### **3. Unified Output**

All processed data is combined into a single CSV file with:

- Consistent column structure
- Proper chronological ordering (when applicable)
- Unified sequence IDs across all databases
- Complete statistics aggregation
- **Schema Compatibility**: The `user_id` column (present in multi-user databases like UoM) is automatically removed to ensure compatibility when combining databases with different structures

### How It Works

1. **Input**: Provide multiple database folders as command-line arguments
2. **Processing**: Each database is processed independently with the same preprocessing pipeline
3. **Schema Normalization**: The `user_id` column is removed from multi-user databases to ensure compatible schemas
4. **ID Tracking**: Sequence IDs are tracked and offset to ensure uniqueness
5. **Combination**: All processed DataFrames are combined into a single output
6. **Statistics**: Statistics are aggregated across all databases for comprehensive reporting

### Example Workflow

```bash
# You have three different databases:
# - ./dexcom_2023/     (Dexcom G6 format, 2023 data)
# - ./libre3_2024/     (Libre 3 format, 2024 data)
# - ./uom_data/        (UoM T1D format, multi-user)

# Combine them all into one ML-ready file:
python glucose_cli.py ./dexcom_2023 ./libre3_2024 ./uom_data --output combined_all_data.csv --verbose

# Output will show:
# - Database 1 (dexcom_2023): Sequences 0-50
# - Database 2 (libre3_2024): Sequences 51-120
# - Database 3 (uom_data): Sequences 121-300
# - Combined: 301 total sequences, all data unified
```

### Statistics Output

When processing multiple databases, the statistics output includes:

- **Multi-Database Info**: List of all processed databases
- **Sequence ID Ranges**: Shows the sequence ID range for each database
- **Aggregated Statistics**: Combined statistics across all databases
- **Individual Database Details**: Separate statistics for each database

### Use Cases

1. **Combining Multiple Patients**: Merge data from multiple patients using different devices
2. **Longitudinal Analysis**: Combine data from the same patient across different time periods
3. **Device Comparison**: Compare and analyze data from different glucose monitoring devices
4. **Research Studies**: Create comprehensive datasets from multiple data sources
5. **ML Training**: Build larger training datasets by combining multiple databases

### Best Practices

1. **Consistent Parameters**: Use the same preprocessing parameters for all databases to ensure consistency
2. **Compatible Formats**: While different formats are supported, ensure data quality is similar across databases
3. **Sequence Length**: Consider using a consistent `min_sequence_len` parameter for all databases
4. **Output Naming**: Use descriptive output file names that indicate multi-database processing
5. **User ID Tracking**: Note that the `user_id` column is automatically removed when combining databases. If you need to track individual users across databases, consider using single-database processing for each user separately

## 📁 Input File Requirements

### Supported Database Types

The system now supports multiple database types with automatic detection and conversion:

#### 1. **Dexcom G6 Database** (`dexcom`)

- **Format**: Single CSV files with Dexcom G6 format
- **Structure**: Mono-user database
- **Files**: `000-14 oct-28 oct 2019.csv`, `001-28 oct-10 nov 2019.csv`, etc.
- **Detection**: Files with numeric prefixes and date ranges in names

#### 2. **FreeStyle Libre 3 Database** (`libre3`)

- **Format**: Libre 3 CSV export format
- **Structure**: Mono-user database
- **Files**: `FreeStyle_Libre_3__11-12-2024.csv`
- **Detection**: Files with "FreeStyle_Libre_3" in name

#### 3. **University of Manchester T1D Database** (`uom`) - **NEW!**

- **Format**: Multi-user database with organized folder structure
- **Structure**: Multi-user database with separate folders for different data types
- **Files**: Organized by data type and user ID
- **Detection**: Files with "UoM" prefix and user ID patterns

### Database Auto-Detection

The system automatically detects the database type based on file patterns and folder structure:

```bash
# Works with any supported database type
python glucose_cli.py ./000-csv        # Dexcom format
python glucose_cli.py ./libre3         # Libre 3 format
python glucose_cli.py ./zendo_small    # UoM T1D format
```

### Folder Structures

#### **Dexcom G6 Format**

```
000-csv/
├── 000-14 oct-28 oct 2019.csv
├── 001-28 oct-10 nov 2019.csv
├── 002-11 nov-24 nov 2019.csv
└── ... (additional CSV files)
```

#### **FreeStyle Libre 3 Format**

```
libre3/
├── FreeStyle_Libre_3__11-12-2024.csv
└── ... (additional Libre 3 files)
```

#### **University of Manchester T1D Format** - **NEW!**

```
zendo_small/
├── Glucose Data/
│   ├── UoMGlucose2301.csv
│   ├── UoMGlucose2302.csv
│   └── ...
├── Insulin Data/
│   ├── Basal Data/
│   │   ├── UoMBasal2301.csv
│   │   └── ...
│   └── Bolus Data/
│       ├── UoMBolus2301.csv
│       └── ...
├── Nutrition Data/
│   ├── UoMNutrition2301.csv
│   └── ...
└── README.md
```

### CSV File Formats

Each database type has its own CSV format, but the system automatically converts them to a standardized format:

#### **Standardized Output Format**

| Column Name                       | Description           | Example                                 |
| --------------------------------- | --------------------- | --------------------------------------- |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | ISO timestamp         | `2019-10-28T16:42:37`                   |
| `Event Type`                      | Type of glucose event | `EGV`, `Calibration`, `Insulin`, `Carb` |
| `Glucose Value (mg/dL)`           | Glucose reading       | `120.0`                                 |
| `Insulin Value (u)`               | Insulin amount        | `2.5`                                   |
| `Carb Value (grams)`              | Carbohydrate amount   | `30.0`                                  |
| `user_id`                         | User identifier       | `2301` (for multi-user databases)       |

#### **Supported Event Types**

- **EGV**: Estimated Glucose Value (main glucose readings)
- **Calibration**: Calibration events (can cause data spikes)
- **Insulin**: Insulin administration
- **Carb**: Carbohydrate intake

## 📊 Processing Information Fields Explained

### Step-by-Step Processing Output

#### 1. Consolidation Phase

```
Found 34 CSV files to consolidate
Processing: 000-14 oct-28 oct 2019.csv
  ✓ Extracted 4,014 records
Total records collected: 1,234,567
Records with valid timestamps: 1,200,000
Date range: 2019-10-14T16:42:37 to 2025-09-17T13:30:12
```

#### 2. High/Low Value Replacement

```
Replaced 245 'High' values with 401
Replaced 89 'Low' values with 39
Total replacements: 334
✓ Glucose field converted to Float64 type
```

#### 3. Calibration Event Removal

```
Found 245 calibration events to remove
Removed 245 calibration events
Records before removal: 1,200,000
Records after removal: 1,199,755
✓ Calibration events removed - gaps can now be interpolated
```

#### 4. Gap Detection and Sequences

```
Created 1,245 sequences
Found 1,244 gaps > 15 minutes
Calibration Periods Detected: 89
Records Removed After Calibration: 45,678
```

#### 5. Interpolation Analysis

```
Small Gaps Identified and Processed: 2,456 gaps
Interpolated Data Points Created: 4,123 points
Total Field Interpolations: 8,246 values
Glucose Interpolations: 4,123 values
Large Gaps Skipped: 1,244 gaps
```

#### 6. Sequence Filtering

```
Original Sequences: 1,245
Sequences After Filtering: 892
Sequences Removed: 353
Original Records: 1,154,322
Records After Filtering: 987,654
```

#### 7. Fixed-Frequency Data Creation

```
Creating fixed-frequency data with 5-minute intervals...
Processed 892 sequences
Time adjustments made: 156
Glucose interpolations: 12,456
Insulin records shifted: 8,234
Carb records shifted: 6,789
Records before: 987,654
Records after: 1,023,456
Fixed-frequency data creation complete
```

### Why More Small Gaps Than Interpolations?

The difference between "Small Gaps Identified" and actual interpolations occurs due to several factors:

#### 1. **Gap Size Thresholds**

- **Small Gap Detection**: Any gap between `expected_interval` (5 min) and `small_gap_max` (15 min)
- **Interpolation Logic**: Only gaps that represent 1-2 missing data points are interpolated

#### Example:

- Gap of 10 minutes detected (2 missing 5-minute intervals)
- Gap of 12 minutes detected (2.4 missing intervals) → Only 2 points interpolated
- Gap of 20 minutes detected → Skipped (exceeds small_gap_max)

#### 2. **Data Quality Requirements**

- Both previous and next values must be valid numeric data
- If either value is missing or non-numeric, interpolation is skipped
- Only glucose, insulin, and carb values are interpolated (not timestamps or event types)

#### 3. **Sequence Boundaries**

- Gaps at sequence boundaries are not interpolated
- Large gaps (>15 min) create new sequences instead of being filled

### Why Calibration Removal Instead of Interpolation?

The new approach removes calibration events entirely rather than interpolating their values because:

1. **Cleaner Data**: Calibration events often contain spikes or inaccurate readings
2. **Better Interpolation**: Removing calibration events creates gaps that can be interpolated using surrounding EGV (glucose) values
3. **Consistent Methodology**: All gaps are treated uniformly in the interpolation step
4. **Reduced Noise**: Eliminates potential artifacts from calibration spikes

### Why Fewer Actual Interpolated Values?

The "Total Field Interpolations" count includes:

- **Glucose Value interpolations**: Most common
- **Insulin Value interpolations**: Only when both values are valid
- **Carb Value interpolations**: Only when both values are valid

**Common scenarios where interpolation is skipped:**

1. **Missing adjacent values**: Previous or next value is empty
2. **Non-numeric values**: Text or invalid data in adjacent cells
3. **Mixed event types**: Different event types around the gap
4. **Boundary conditions**: Gaps at the start/end of sequences

### High/Low Value Replacement

The system automatically replaces textual glucose values with numeric equivalents:

- **"High"** → **401 mg/dL** (configurable): Represents glucose readings above the device's upper measurement limit
- **"Low"** → **39 mg/dL** (configurable): Represents glucose readings below the device's lower measurement limit

#### Configurable Replacement Values

You can customize the High/Low replacement values through:

1. **Configuration File** (`glucose_config.yaml`):

   ```yaml
   glucose_value_replacement:
     high_value: 401 # Custom high value
     low_value: 39 # Custom low value
     enabled: true
   ```

2. **Command Line** (when using config file):
   ```bash
   python glucose_cli.py ./000-csv --config glucose_config.yaml --high-value 450 --low-value 30
   ```

#### Why This Replacement is Necessary

1. **ML Compatibility**: Machine learning models require numeric data
2. **Consistent Data Types**: Ensures the glucose field is Float64 throughout the dataset
3. **Meaningful Values**: 401 and 39 are clinically relevant threshold values that preserve the information that readings were outside normal measurement ranges
4. **Flexibility**: Allows customization for different devices or clinical requirements

The replacement occurs early in the pipeline (Step 2) to ensure all subsequent processing works with numeric glucose values.

### Fixed-Frequency Data Creation (Step 7)

The fixed-frequency step is a new feature that creates consistent time intervals for machine learning applications:

#### **What It Does**

1. **Time Alignment**: Aligns the first point of each sequence to the nearest round minute
2. **Fixed Intervals**: Creates timestamps at exactly the specified interval (default: 5 minutes)
3. **Glucose Interpolation**: **IMPROVED** - Ensures every row has a glucose value by interpolating from the nearest valid glucose readings, regardless of event type
4. **Event Shifting**: Shifts carb and insulin values to the closest datapoints (preserves discrete events)
5. **Quality Assurance**: Guarantees no empty glucose values in the final output

#### **Why It's Important for ML**

- **Consistent Structure**: All sequences have exactly the same time intervals
- **Predictable Format**: ML models can expect data points at regular intervals
- **Better Interpolation**: Glucose values are properly interpolated rather than just shifted
- **Preserved Events**: Carb and insulin events are preserved by shifting to closest points

#### **Example Transformation**

**Before (irregular intervals):**

```
10:02:30 - Glucose: 100.0
10:08:45 - Glucose: 110.0
10:15:15 - Glucose: 120.0
```

**After (fixed 5-minute intervals):**

```
10:00:00 - Glucose: 100.0
10:05:00 - Glucose: 110.0 (interpolated)
10:10:00 - Glucose: 120.0
10:15:00 - Glucose: 130.0 (interpolated)
```

#### **Configuration Options**

- **Enable/Disable**: Use `--fixed-frequency/--no-fixed-frequency` or `create_fixed_frequency: true/false`
- **Default**: Enabled by default (recommended for ML applications)
- **Interval**: Controlled by `--interval` parameter (default: 5 minutes)

## 🔧 Advanced Configuration

### Calibration Period Detection

The system can automatically detect calibration periods (typically 2-3 hours) and remove subsequent data to avoid inaccurate readings:

```bash
python glucose_cli.py ./000-csv --calibration-period 165 --remove-after-calibration 24
```

### Custom Interpolation Settings

Fine-tune interpolation behavior:

```bash
# More aggressive interpolation (up to 30-minute gaps)
python glucose_cli.py ./000-csv --gap-max 30 --interval 5

# Conservative interpolation (only 10-minute gaps)
python glucose_cli.py ./000-csv --gap-max 10 --interval 5
```

### Debugging and Analysis

Enable intermediate file saving to inspect each processing step:

```bash
python glucose_cli.py ./000-csv --save-intermediate --verbose
```

This creates files like:

- `consolidated_data.csv`
- `step2_high_low_replaced.csv`
- `step3_calibrations_removed.csv`
- `step4_sequences_created.csv`
- `step5_interpolated_values.csv`
- `step6_filtered_sequences.csv`
- `step7_fixed_frequency.csv`
- `step8_glucose_only.csv`
- `step9_ml_ready.csv`

## 📈 Output Data Format

### Standard Output Format

The final ML-ready dataset contains:

| Column                            | Type    | Description                                    |
| --------------------------------- | ------- | ---------------------------------------------- |
| `sequence_id`                     | Integer | Unique identifier for continuous data segments |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | String  | ISO formatted timestamp                        |
| `Event Type`                      | String  | Type of glucose event                          |
| `Glucose Value (mg/dL)`           | Float64 | Glucose reading (interpolated where needed)    |
| `Insulin Value (u)`               | Float64 | Insulin amount                                 |
| `Carb Value (grams)`              | Float64 | Carbohydrate amount                            |

### Glucose-Only Output Format

When using the `--glucose-only` option, the output is simplified to contain only glucose data:

| Column                            | Type    | Description                                    |
| --------------------------------- | ------- | ---------------------------------------------- |
| `sequence_id`                     | Integer | Unique identifier for continuous data segments |
| `Timestamp (YYYY-MM-DDThh:mm:ss)` | String  | ISO formatted timestamp                        |
| `Glucose Value (mg/dL)`           | Float64 | Glucose reading (interpolated where needed)    |

#### Key Differences in Glucose-Only Mode:

- **Removed Fields**: `Event Type`, `Insulin Value (u)`, `Carb Value (grams)`
- **Filtered Data**: Only rows with non-null glucose values are included
- **Simplified Output**: Focuses exclusively on glucose time-series data
- **Smaller File Size**: Reduced data volume for glucose-specific analysis

#### Example Output Comparison:

**Standard Format:**

```csv
sequence_id,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams)
0,2019-10-14T16:47:37,EGV,55.0,,
0,2019-10-14T16:52:37,EGV,55.0,,
```

**Glucose-Only Format:**

```csv
sequence_id,Timestamp (YYYY-MM-DDThh:mm:ss),Glucose Value (mg/dL)
0,2019-10-14T16:47:37,55.0
0,2019-10-14T16:52:37,55.0
```

## 🎯 Machine Learning Applications

The processed data is optimized for:

- **Time Series Forecasting**: Predict future glucose levels
- **Anomaly Detection**: Identify unusual glucose patterns
- **Sequence Modeling**: LSTM, GRU, or Transformer models
- **Classification**: Event type prediction or risk assessment

### When to Use Glucose-Only Mode

The `--glucose-only` option is particularly useful for:

- **Pure Glucose Analysis**: When you only need glucose time-series data
- **Simplified Models**: For models that don't require insulin or carb information
- **Data Reduction**: When working with large datasets and need to reduce file size
- **Glucose Forecasting**: For models focused exclusively on glucose prediction
- **Research Applications**: When studying glucose patterns without confounding variables

### Use Cases for Each Format

**Standard Format** (all fields):

- Multi-variate time series analysis
- Models that consider insulin and carb effects
- Comprehensive diabetes management applications
- Research requiring full context

**Glucose-Only Format**:

- Pure glucose forecasting models
- Anomaly detection in glucose patterns
- Simplified time series analysis
- Educational or demonstration purposes

## 🔧 Converter Architecture

### Overview

The system uses a modular converter architecture that automatically detects and processes different database formats. This allows support for multiple glucose monitoring devices and database structures without manual configuration.

### Architecture Components

#### 1. **Database Detection System**

- **`DatabaseDetector`**: Automatically identifies database type based on file patterns
- **Supported Types**: Dexcom G6, FreeStyle Libre 3, University of Manchester T1D
- **Detection Logic**: Analyzes file names, folder structure, and header patterns

#### 2. **Database Converters**

- **`DatabaseConverter`**: Abstract base class for all database converters
- **`MonoUserDatabaseConverter`**: Handles single-user databases (Dexcom, Libre 3)
- **`MultiUserDatabaseConverter`**: Handles multi-user databases (UoM T1D)

#### 3. **Format Detection System**

- **`CSVFormatDetector`**: Detects individual CSV file formats within databases
- **Format Converters**: Specialized converters for each data type (glucose, insulin, nutrition, etc.)
- **Auto-Detection**: Automatically identifies file format based on column headers

### Supported Database Types

#### **Mono-User Databases**

- **Dexcom G6**: Single user, multiple CSV files with date ranges
- **FreeStyle Libre 3**: Single user, Libre 3 export format

#### **Multi-User Databases**

- **University of Manchester T1D**: Multiple users, organized folder structure by data type

### Adding New Database Types

To add support for a new database type:

1. **Create Database Converter**: Extend `DatabaseConverter` class
2. **Implement Detection Logic**: Add file pattern detection in `DatabaseDetector`
3. **Create Format Converters**: Add specialized CSV format converters if needed
4. **Register Converter**: Add to the `database_converters` dictionary

### Benefits of This Architecture

- **Extensibility**: Easy to add new database types
- **Maintainability**: Clear separation of concerns
- **Flexibility**: Handles both mono-user and multi-user databases
- **Auto-Detection**: No manual configuration required
- **Standardization**: All databases converted to consistent format

## 📊 University of Manchester T1D Dataset (Zendo)

### Overview

The **T1D-UOM (Type 1 Diabetes - University of Manchester)** dataset is a comprehensive longitudinal multimodal dataset containing Type 1 Diabetes data from 16 individuals collected from October 2023 to August 2024.

### Dataset Features

- **Multi-Modal Data**: Glucose, insulin, nutrition, activity, and sleep data
- **Longitudinal**: Up to 10+ months of continuous monitoring per participant
- **High Frequency**: Glucose readings every 5 minutes, activity data every minute
- **Comprehensive**: Includes both medical and lifestyle factors
- **Ethically Approved**: University of Manchester Ethical Approval 2024-15687-33719

### Data Structure

#### **Glucose Data** (`UoMGlucose*.csv`)

- **Format**: `bg_ts` (datetime), `value` (mmol/L)
- **Frequency**: Every 5 minutes
- **Coverage**: 356,146 records across 17 files
- **Units**: Blood glucose in mmol/L (automatically converted to mg/dL)

#### **Insulin Data**

- **Basal Insulin** (`UoMBasal*.csv`): `basal_ts`, `basal_dose`, `insulin_kind`
- **Bolus Insulin** (`UoMBolus*.csv`): `bolus_ts`, `bolus_dose`
- **Coverage**: 20,407 basal records, 5,660 bolus records
- **Units**: Insulin units (U) and units per hour (U/h)

#### **Nutrition Data** (`UoMNutrition*.csv`)

- **Format**: `meal_ts`, `meal_type`, `carbs_g`, `prot_g`, `fat_g`, `fibre_g`
- **Meal Types**: Breakfast, Lunch, Dinner, Snack
- **Coverage**: 4,351 nutrition records
- **Units**: Grams for all macronutrients

#### **Activity Data** (`UoMActivity*.csv`)

- **Format**: `activity_ts`, `activity_type`, `step_count`, `distance_m`, `met`
- **Types**: SEDENTARY, WALKING, RUNNING, GENERIC
- **Coverage**: 228,681 activity records
- **Metrics**: Steps, distance, calories, MET values

#### **Sleep Data** (`UoMSleep*.csv`)

- **Format**: `sleep_ts`, `heart_rate`, `sleep_level`, `stress_level_value`
- **Coverage**: 323,340 sleep records
- **Metrics**: Heart rate, sleep stages, stress levels

### Usage with the Preprocessor

The dataset is fully compatible with the preprocessing system:

```bash
# Process the full zendo dataset
python glucose_cli.py ./zenodo_archive

# Process the smaller test dataset
python glucose_cli.py ./zendo_small

# Process with custom settings
python glucose_cli.py ./zendo_small --interval 5 --min-length 100 --verbose
```

### Key Advantages

1. **Multi-User Support**: Handles 16 different participants automatically
2. **Data Integration**: Combines glucose, insulin, nutrition, and activity data
3. **High Quality**: Ethically approved, research-grade dataset
4. **Comprehensive**: Covers all aspects of diabetes management
5. **Longitudinal**: Long-term data for trend analysis and prediction

### Citation

If you use this dataset, please cite:

```bibtex
@dataset{t1d_uom_2024,
  title={T1D-UOM – A Longitudinal Multimodal Dataset of Type 1 Diabetes},
  author={University of Manchester},
  year={2024},
  doi={10.5281/zenodo.15169263},
  url={https://doi.org/10.5281/zenodo.15169263}
}
```

### Dataset Access

- **Full Dataset**: Available on [Zenodo](https://doi.org/10.5281/zenodo.15169263)
- **Test Dataset**: `zendo_small/` folder (subset for testing)
- **Documentation**: Complete data dictionary in `zendo_small/README.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🆕 Recent Updates

### Version 2.1 - Multi-Database Processing (NEW!)

#### **Multi-Database Support**

- **Process Multiple Databases**: Combine data from multiple database folders into a single output file
- **Consistent ID Tracking**: Automatic sequence ID tracking and offsetting across databases
- **Format-Agnostic**: Mix different database formats (Dexcom, Libre3, UoM) in one command
- **Aggregated Statistics**: Comprehensive statistics across all processed databases
- **CLI Enhancement**: Simple command-line interface for multi-database processing

### Version 2.0 - Major Architecture Overhaul

#### **New Converter System**

- **Modular Architecture**: Complete rewrite with pluggable database converters
- **Auto-Detection**: Automatic database type detection based on file patterns
- **Multi-Database Support**: Dexcom G6, FreeStyle Libre 3, University of Manchester T1D
- **Extensible Design**: Easy to add new database types and formats

#### **Enhanced Data Processing**

- **Fixed Glucose Interpolation**: **CRITICAL FIX** - Ensures every row has a glucose value in fixed-frequency data
- **Improved Error Handling**: Robust statistics handling with proper null checks
- **Multi-User Support**: Full support for multi-user databases with user ID tracking
- **Better Data Quality**: Guaranteed glucose values in all output rows

#### **New Dataset Support**

- **University of Manchester T1D Dataset**: Full support for the comprehensive T1D-UOM dataset
- **Multi-Modal Data**: Glucose, insulin, nutrition, activity, and sleep data integration
- **Longitudinal Analysis**: Support for long-term data analysis across multiple participants

#### **Bug Fixes**

- **KeyError Fixes**: Resolved statistics KeyError issues in CLI and preprocessor
- **Glucose Interpolation**: Fixed missing glucose values in non-EGV event rows
- **Statistics Robustness**: Added proper null checks and default values throughout

### Migration Guide

If upgrading from a previous version:

1. **Database Detection**: The system now automatically detects database types - no manual configuration needed
2. **Output Format**: Output format remains the same, but now includes `user_id` for multi-user databases
3. **Configuration**: Existing configuration files remain compatible
4. **CLI Usage**: No changes to command-line interface - just point to your data folder

### Breaking Changes

- **None**: All existing functionality remains compatible
- **Enhanced**: All features are enhanced with better error handling and data quality

## 📄 License

[Add your license information here]

## 🆘 Support

For issues or questions:

1. Check the verbose output with `--verbose` flag
2. Enable intermediate file saving with `--save-intermediate`
3. Review the processing statistics for data quality insights
4. Open an issue on GitHub with sample data and error messages
