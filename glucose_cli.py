#!/usr/bin/env python3
"""
Command Line Interface for Glucose Data Preprocessing

This script provides a CLI wrapper around the glucose preprocessing functionality,
allowing users to process glucose data from CSV folders through command line arguments.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import typer
import polars as pl
from loguru import logger
from glucose_ml_preprocessor import GlucoseMLPreprocessor
from processing.stats_manager import print_statistics as sm_print_statistics

# Optional cycle data parser (research-phase convenience)
try:
    from cycle_data_parser import CycleDataParser
except ImportError:
    CycleDataParser = None

app = typer.Typer(help="Glucose Data Preprocessing CLI")

def _generate_output_filename(input_folders: List[Path], user_output: Path) -> Path:
    """
    Generate output filename based on source folder names.
    
    If user provided a custom output path, use that name but place it in OUTPUT folder.
    Otherwise, generate filename from source folder names.
    """
    output_dir = Path("OUTPUT")
    output_dir.mkdir(exist_ok=True)
    
    # If user provided a custom output path, use that filename
    if user_output != Path("glucose_ml_ready.csv"):
        return output_dir / user_output.name
    
    # Generate filename from source folder names
    source_names: List[str] = []
    for folder in input_folders:
        if folder.is_file() and folder.suffix.lower() == ".zip":
            # For zip files, use the filename without extension
            source_names.append(folder.stem)
        else:
            # For folders, use the folder name
            source_names.append(folder.name)
    
    # Sanitize names (remove invalid characters for filenames)
    sanitized_names = []
    invalid_chars = '<>:"/\\|?*'
    for name in source_names:
        sanitized = ''.join(c if c not in invalid_chars else '_' for c in name)
        sanitized_names.append(sanitized)
    
    # Create filename: combine source names with underscores
    if len(sanitized_names) == 1:
        filename = f"{sanitized_names[0]}_ml_ready.csv"
    else:
        # For multiple sources, combine them
        combined = "_".join(sanitized_names)
        filename = f"{combined}_ml_ready.csv"
    
    return output_dir / filename

def _resolve_config_file(config_file: Optional[Path]) -> Optional[Path]:
    """
    Resolve the config file path to use.

    Behavior:
    - If user provides --config, use it as-is.
    - Otherwise, if ./glucose_config.yaml exists, use it by default.
    - Otherwise, return None and rely on CLI args only.
    """
    if config_file:
        return config_file
    default_cfg = Path("glucose_config.yaml")
    if default_cfg.exists() and default_cfg.is_file():
        return default_cfg
    return None

@app.command()
def main(
    input_folders: List[Path] = typer.Argument(
        ..., 
        help="Path(s) to input datasets to process. Each input can be a folder (CSV datasets) or a .zip (AI-READI). Multiple inputs can be combined."
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file (command line args override config values)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file name (will be saved in OUTPUT folder). If not provided, filename is generated from source folder names."
    ),
    interval_minutes: int = typer.Option(
        5,
        "--interval", "-i",
        help="Time discretization interval in minutes"
    ),
    gap_max_minutes: int = typer.Option(
        15,
        "--gap-max", "-g",
        help="Maximum gap size to interpolate in minutes"
    ),
    min_sequence_len: int = typer.Option(
        200,
        "--min-length", "-l",
        help="Minimum sequence length to keep for ML training"
    ),
    remove_calibration: bool = typer.Option(
        True,
        "--remove-calibration/--keep-calibration",
        help="Remove calibration events to create interpolatable gaps"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    show_stats: bool = typer.Option(
        True,
        "--stats/--no-stats",
        help="Show processing statistics"
    ),
    save_intermediate_files: bool = typer.Option(
        False,
        "--save-intermediate", "-s",
        help="Save intermediate files after each processing step"
    ),
    calibration_period_minutes: int = typer.Option(
        60*2 + 45,  # 2 hours 45 minutes
        "--calibration-period", "-p",
        help="Gap duration considered as calibration period in minutes (default: 165 minutes)"
    ),
    remove_after_calibration_hours: int = typer.Option(
        24,
        "--remove-after-calibration", "-r",
        help="Hours of data to remove after calibration period (default: 24 hours)"
    ),
    glucose_only: bool = typer.Option(
        False,
        "--glucose-only",
        help="Output only glucose data: remove Event Type, Insulin Value, and Carb Value fields, keep only rows with glucose values"
    ),
    create_fixed_frequency: bool = typer.Option(
        True,
        "--fixed-frequency/--no-fixed-frequency",
        help="Create fixed-frequency data with consistent intervals (default: enabled)"
    ),
    cycle_data_file: Optional[Path] = typer.Option(
        None,
        "--cycle",
        help="Path to cycle data CSV file with columns: date, flow_amount. Glucose data will be filtered to cycle data date range."
    ),
    first_n_users: Optional[int] = typer.Option(
        None,
        "--first-n-users",
        help="Process only the first n users (for multi-user databases). If 0 or not specified, all users are processed."
    )
) -> None:
    """
    Process glucose data from CSV folder(s) for machine learning.
    
    This tool consolidates CSV files from one or more input folders and processes them through
    the complete ML preprocessing pipeline including High/Low value replacement, calibration removal, 
    gap detection, interpolation, calibration period detection, and sequence filtering.
    
    When multiple folders are provided, sequence IDs are tracked and offset to ensure consistency
    across different databases, producing a single unified output file.
    """
    # Configure loguru to show only the message, matching the old print() behavior
    logger.remove()
    logger.add(sys.stdout, format="{message}")
    
    # Validate input paths
    validated_folders: List[Path] = []
    for input_folder in input_folders:
        if not input_folder.exists():
            logger.error(f"Error: Input path '{input_folder}' does not exist")
            raise typer.Exit(1)
        
        if not input_folder.is_dir():
            # Allow zip-backed datasets (AI-READI)
            if not (input_folder.is_file() and input_folder.suffix.lower() == ".zip"):
                logger.error(
                    f"Error: Input must be a folder containing CSV files or a .zip dataset, got: '{input_folder}'"
                )
                raise typer.Exit(1)
        
        validated_folders.append(input_folder)
    
    # Generate output filename and ensure OUTPUT folder exists
    if output_file is None:
        output_file = Path("glucose_ml_ready.csv")  # Default for filename generation
    final_output_file = _generate_output_filename(validated_folders, output_file)
    
    # Initialize preprocessor info
    if verbose:
        logger.info("Initializing glucose data preprocessor...")
        if len(validated_folders) == 1:
            logger.info(f"   Input folder: {validated_folders[0]}")
        else:
            logger.info(f"   Input folders ({len(validated_folders)}):")
            for i, folder in enumerate(validated_folders, 1):
                logger.info(f"      {i}. {folder}")
        logger.info(f"   Output file: {final_output_file}")
        logger.info(f"   Time interval: {interval_minutes} minutes")
        logger.info(f"   Gap max: {gap_max_minutes} minutes")
        logger.info(f"   Min sequence length: {min_sequence_len}")
        logger.info(f"   Remove calibration events: {remove_calibration}")
        logger.info(f"   Calibration period: {calibration_period_minutes} minutes")
        logger.info(f"   Remove after calibration: {remove_after_calibration_hours} hours")
        logger.info(f"   Save intermediate files: {save_intermediate_files}")
        logger.info(f"   Glucose only mode: {glucose_only}")
        logger.info(f"   Fixed-frequency data: {create_fixed_frequency}")
        if cycle_data_file:
            logger.info(f"   Cycle data file: {cycle_data_file}")
    
    try:
        resolved_config_file = _resolve_config_file(config_file)
        if verbose and resolved_config_file and not config_file:
            logger.info(f"Auto-loading default configuration from: {resolved_config_file}")

        # CLI arguments override config file values
        cli_overrides: Dict[str, Any] = {
            'expected_interval_minutes': interval_minutes,
            'small_gap_max_minutes': gap_max_minutes,
            'remove_calibration': remove_calibration,
            'min_sequence_len': min_sequence_len,
            'save_intermediate_files': save_intermediate_files,
            'calibration_period_minutes': calibration_period_minutes,
            'remove_after_calibration_hours': remove_after_calibration_hours,
            'glucose_only': glucose_only,
            'create_fixed_frequency': create_fixed_frequency,
            'first_n_users': first_n_users if first_n_users and first_n_users > 0 else None,
            'verbose': verbose
        }

        # Create preprocessor
        if resolved_config_file:
            if verbose:
                logger.info(f"Loading configuration from: {resolved_config_file}")
            preprocessor = GlucoseMLPreprocessor.from_config_file(resolved_config_file, **cli_overrides)
        else:
            # Use CLI arguments directly
            preprocessor = GlucoseMLPreprocessor(**cli_overrides)
        
        # Resolve output file: CLI > Config > Default
        final_output_file = output_file
        if final_output_file is None:
            final_output_file = preprocessor.output_file
        if final_output_file is None:
            final_output_file = Path("glucose_ml_ready.csv")
            
        # Ensure output directory exists
        if final_output_file.parent and not final_output_file.parent.exists():
            if verbose:
                logger.info(f"Creating output directory: {final_output_file.parent}")
            final_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Parse cycle data if provided
        cycle_parser = None
        if cycle_data_file:
            if CycleDataParser is None:
                logger.error("Error: cycle_data_parser module not found. Cannot process --cycle.")
                raise typer.Exit(1)
            if not cycle_data_file.exists():
                logger.error(f"Error: Cycle data file '{cycle_data_file}' does not exist")
                raise typer.Exit(1)
            
            if verbose:
                logger.info("Parsing cycle data...")
            
            cycle_parser = CycleDataParser()
            cycle_parser.parse_cycle_file(cycle_data_file)
        
        # Process data
        if verbose:
            if len(validated_folders) == 1:
                logger.info("Starting glucose data processing pipeline...")
            else:
                logger.info(f"Starting multi-database processing pipeline for {len(validated_folders)} databases...")
        
        # Process single or multiple databases
        if len(validated_folders) == 1:
            ml_data, statistics, _ = preprocessor.process(
                validated_folders[0], final_output_file
            )
        else:
            ml_data, statistics, _ = preprocessor.process_multiple_databases(
                validated_folders, final_output_file
            )
        
        # Merge cycle data if provided
        if cycle_parser:
            if verbose:
                logger.info("Integrating cycle data...")
            ml_data = preprocessor.merge_cycle_data(ml_data, cycle_parser)
            
            # Save the final data with cycle information
            if final_output_file:
                ml_data.write_csv(final_output_file)
                if verbose:
                    logger.info(f"Final data with cycle information saved to: {final_output_file}")
        
        # Show results
        logger.info(f"Processing completed successfully!")
        
        # Extract record and sequence counts safely
        try:
            overview = statistics.get("dataset_overview", {}) if isinstance(statistics, dict) else {}
            stats_records = int(overview.get("total_records", 0))
            stats_sequences = int(overview.get("total_sequences", 0))
        except (ValueError, TypeError, AttributeError):
            stats_records = 0
            stats_sequences = 0
        
        df_records = len(ml_data) if hasattr(ml_data, "__len__") else 0
        df_sequences = 0
        if isinstance(ml_data, pl.DataFrame) and "sequence_id" in ml_data.columns:
            df_sequences = int(ml_data["sequence_id"].n_unique())

        out_records = stats_records if (df_records == 0 and stats_records > 0) else df_records
        out_sequences = stats_sequences if (df_sequences == 0 and stats_sequences > 0) else df_sequences

        logger.info(f"Output: {out_records:,} records in {out_sequences:,} sequences")
        logger.info(f"Saved to: {final_output_file}")
        
        # Show statistics if requested
        if show_stats:
            if verbose:
                params = {
                    'expected_interval_minutes': preprocessor.expected_interval_minutes,
                    'small_gap_max_minutes': preprocessor.small_gap_max_minutes,
                    'remove_calibration': preprocessor.remove_calibration,
                    'min_sequence_len': preprocessor.min_sequence_len,
                    'calibration_period_minutes': preprocessor.calibration_period_minutes,
                    'remove_after_calibration_hours': preprocessor.remove_after_calibration_hours,
                    'create_fixed_frequency': preprocessor.create_fixed_frequency
                }
                sm_print_statistics(statistics, params)
            else:
                # Show summary statistics only
                overview = statistics.get('dataset_overview', {})
                seq_analysis = statistics.get('sequence_analysis', {})
                logger.info(f"\nSummary:")
                
                date_range = overview.get('date_range', {})
                logger.info(f"   Date range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
                logger.info(f"   Longest sequence: {seq_analysis.get('longest_sequence', 0):,} records")
                
                seq_lengths = seq_analysis.get('sequence_lengths', {})
                logger.info(f"   Average sequence: {seq_lengths.get('mean', 0.0):.1f} records")
                
                # Show data preservation percentage
                original_recs = overview.get('original_records', overview.get('total_records', 0))
                final_recs = overview.get('total_records', 0)
                preservation_pct = (final_recs / original_recs * 100) if original_recs > 0 else 100
                logger.info(f"   Data preserved: {preservation_pct:.1f}% ({final_recs:,}/{original_recs:,} records)")
                
                # Show interpolation summary
                interp_analysis = statistics['interpolation_analysis']
                logger.info(f"   Gaps processed: {interp_analysis.get('small_gaps_filled', 0):,} gaps")
                logger.info(f"   Data points created: {interp_analysis.get('total_interpolated_data_points', 0):,} points")
                logger.info(f"   Field interpolations: {interp_analysis.get('total_interpolations', 0):,} values")
                
                # Show filtering summary
                if 'filtering_analysis' in statistics:
                    filter_analysis = statistics['filtering_analysis']
                    logger.info(f"   Sequences filtered: {filter_analysis.get('removed_sequences', 0):,} removed")
                
                # Show calibration period summary
                gap_analysis = statistics.get('gap_analysis', {})
                if 'calibration_period_analysis' in gap_analysis:
                    calib_analysis = gap_analysis['calibration_period_analysis']
                    if calib_analysis.get('calibration_periods_detected', 0) > 0:
                        logger.info(f"   Calibration periods: {calib_analysis['calibration_periods_detected']:,} detected, {calib_analysis.get('total_records_marked_for_removal', 0):,} records removed")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if verbose:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()