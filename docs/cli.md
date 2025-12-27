# Command Line Interface (CLI)

The `glucose_cli.py` script provides a robust interface for processing one or multiple glucose databases into a unified ML-ready format.

## Usage

```bash
uv run glucose_cli.py [INPUT_FOLDERS]... [OPTIONS]
```

### Arguments

- `INPUT_FOLDERS`: One or more paths to database folders (e.g., `data/uom_small`) or ZIP files (for AI-READY).

### Options

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `--config` | `-c` | Path to a YAML configuration file. |
| `--output` | `-o` | Filename for the final ML-ready CSV. |
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
uv run glucose_cli.py data/uom_small data/dexcom_user1 -o combined_data.csv
```

The preprocessor automatically:
1. Detects the database type for each input.
2. Tracks global `sequence_id` to prevent collisions.
3. Normalizes all data to the same time resolution and field set.

## Processing Statistics

At the end of a successful run, the CLI displays a summary including:
- Total records collected and preserved.
- Number of sequences created and filtered.
- Interpolation and gap statistics.
- Longest and average sequence lengths.

