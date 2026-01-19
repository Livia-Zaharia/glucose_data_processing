# Command Line Interface (CLI)

The project provides two main CLI tools:
- `glucose-process`: Processes one or multiple glucose databases into a unified ML-ready format.
- `glucose-compare`: Compares two checkpoint CSV files and provides detailed statistics.

## Usage: glucose-process

```bash
glucose-process [INPUT_FOLDERS]... [OPTIONS]
```

### Arguments

- `INPUT_FOLDERS`: One or more paths to database folders (e.g., `DATA/uom_small`) or ZIP files (for AI-READY).

### Options

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--config` | `-c` | Path to a YAML configuration file. |
| `--output" | `-o` | Filename for the final ML-ready CSV. |
| `--interval` | `-i` | Time discretization interval (minutes). |
| `--gap-max` | `-g` | Max gap size to interpolate (minutes). |
| `--min-length` | `-l` | Minimum sequence length to preserve. |
| `--glucose-only` | | Filter output to only include glucose values. |
| `--no-fixed-frequency` | | Disable resampling to fixed time buckets. |
| `--verbose` | `-v` | Enable detailed logging. |
| `--no-stats` | | Disable the summary statistics printout. |
| `--save-intermediate`| `-s` | Export CSVs after each processing stage. |
| `--first-n-users` | | Limit processing to the first `N` users found. |

## Multi-Database Processing

The CLI supports combining different databases in a single run:

```bash
glucose-process data/uom_small data/dexcom_user1 -o combined_data.csv
```

The preprocessor automatically:
1. Detects the database type for each input.
2. Tracks global `sequence_id` to prevent collisions.
3. Normalizes all data to the same time resolution and field set.

## Comparison Tool: glucose-compare

```bash
glucose-compare [FILE1] [FILE2] [OPTIONS]
```

This tool compares two checkpoint files to ensure processing results are consistent.

### Arguments

- `FILE1`: Path to the first checkpoint file.
- `FILE2`: Path to the second checkpoint file.

### Options

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--key-columns` | `-k` | Key columns for row matching. |
| `--tolerance` | `-t` | Numeric tolerance for approximate matches. |
| `--no-streaming`| | Disable Polars streaming. |

## Processing Statistics

At the end of a successful run, the CLI displays a summary including:
- Total records collected and preserved.
- Number of sequences created and filtered.
- Interpolation and gap statistics.
- Longest and average sequence lengths.

