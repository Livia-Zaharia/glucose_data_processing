#!/usr/bin/env python3
"""
Compare two checkpoint CSV files and provide detailed statistics.

This script compares:
- Field number, names, and order
- Sequence IDs and their row counts
- Row values (exact and approximate matches)
- Overall statistics
"""

import typer
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

app = typer.Typer()


def compare_schemas(df1: pl.DataFrame, df2: pl.DataFrame) -> Dict[str, Any]:
    """Compare schemas of two DataFrames."""
    schema1 = df1.schema
    schema2 = df2.schema
    
    result = {
        'field_count_match': len(schema1) == len(schema2),
        'field_count_1': len(schema1),
        'field_count_2': len(schema2),
        'field_names_match': list(schema1.keys()) == list(schema2.keys()),
        'field_names_1': list(schema1.keys()),
        'field_names_2': list(schema2.keys()),
        'field_types_match': schema1 == schema2,
        'missing_in_2': [col for col in schema1.keys() if col not in schema2],
        'missing_in_1': [col for col in schema2.keys() if col not in schema1],
        'type_differences': {}
    }
    
    # Check for type differences in common columns
    common_cols = set(schema1.keys()) & set(schema2.keys())
    for col in common_cols:
        if schema1[col] != schema2[col]:
            result['type_differences'][col] = {
                'type_1': str(schema1[col]),
                'type_2': str(schema2[col])
            }
    
    return result


def compare_sequences(df1: pl.DataFrame, df2: pl.DataFrame) -> Dict[str, Any]:
    """Compare sequence IDs and their row counts."""
    # Get sequence statistics for both DataFrames
    seq1_stats = df1.group_by('sequence_id').agg([
        pl.len().alias('row_count')
    ]).sort('sequence_id')
    
    seq2_stats = df2.group_by('sequence_id').agg([
        pl.len().alias('row_count')
    ]).sort('sequence_id')
    
    # Compare sequences
    seq1_ids = set(seq1_stats['sequence_id'].to_list())
    seq2_ids = set(seq2_stats['sequence_id'].to_list())
    
    common_seqs = seq1_ids & seq2_ids
    only_in_1 = seq1_ids - seq2_ids
    only_in_2 = seq2_ids - seq1_ids
    
    # Compare row counts for common sequences
    seq1_dict = dict(zip(seq1_stats['sequence_id'].to_list(), seq1_stats['row_count'].to_list()))
    seq2_dict = dict(zip(seq2_stats['sequence_id'].to_list(), seq2_stats['row_count'].to_list()))
    
    row_count_matches = 0
    row_count_differences = []
    
    for seq_id in common_seqs:
        count1 = seq1_dict[seq_id]
        count2 = seq2_dict[seq_id]
        if count1 == count2:
            row_count_matches += 1
        else:
            row_count_differences.append({
                'sequence_id': seq_id,
                'count_1': count1,
                'count_2': count2,
                'difference': count2 - count1
            })
    
    return {
        'total_sequences_1': len(seq1_ids),
        'total_sequences_2': len(seq2_ids),
        'common_sequences': len(common_seqs),
        'only_in_1': len(only_in_1),
        'only_in_2': len(only_in_2),
        'sequence_ids_match': seq1_ids == seq2_ids,
        'row_count_matches': row_count_matches,
        'row_count_differences': len(row_count_differences),
        'row_count_differences_details': row_count_differences[:20],  # Limit to first 20
        'only_in_1_ids': sorted(list(only_in_1))[:20] if only_in_1 else [],
        'only_in_2_ids': sorted(list(only_in_2))[:20] if only_in_2 else []
    }


def compare_values(df1: pl.DataFrame, df2: pl.DataFrame, 
                   key_cols: List[str] = None) -> Dict[str, Any]:
    """Compare row values between two DataFrames."""
    if key_cols is None:
        # Default key columns for comparison
        key_cols = ['sequence_id', 'Timestamp (YYYY-MM-DDThh:mm:ss)']
        # Filter to columns that exist in both DataFrames
        key_cols = [col for col in key_cols if col in df1.columns and col in df2.columns]
    
    if not key_cols:
        return {
            'error': 'No common key columns found for comparison'
        }
    
    # Sort both DataFrames by key columns
    df1_sorted = df1.sort(key_cols)
    df2_sorted = df2.sort(key_cols)
    
    # Join on key columns
    joined = df1_sorted.join(
        df2_sorted,
        on=key_cols,
        how='full',
        suffix='_2'
    )
    
    # Count matches
    total_rows_1 = len(df1_sorted)
    total_rows_2 = len(df2_sorted)
    matched_rows = len(joined.filter(
        pl.all_horizontal([pl.col(col).is_not_null() & pl.col(f'{col}_2').is_not_null() 
                          for col in key_cols])
    ))
    only_in_1 = len(joined.filter(
        pl.all_horizontal([pl.col(col).is_not_null() for col in key_cols]) &
        pl.any_horizontal([pl.col(f'{col}_2').is_null() for col in key_cols])
    ))
    only_in_2 = len(joined.filter(
        pl.any_horizontal([pl.col(col).is_null() for col in key_cols]) &
        pl.all_horizontal([pl.col(f'{col}_2').is_not_null() for col in key_cols])
    ))
    
    # Compare values for matched rows
    value_comparisons = {}
    common_cols = [col for col in df1.columns if col in df2.columns and col not in key_cols]
    
    for col in common_cols:
        if col in joined.columns and f'{col}_2' in joined.columns:
            # Filter to matched rows only
            matched = joined.filter(
                pl.all_horizontal([pl.col(k).is_not_null() & pl.col(f'{k}_2').is_not_null() 
                                  for k in key_cols])
            )
            
            if len(matched) == 0:
                continue
            
            col1_vals = matched[col]
            col2_vals = matched[f'{col}_2']
            
            # Check if numeric
            is_numeric = col1_vals.dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
            
            if is_numeric:
                # Numeric comparison
                exact_matches = (col1_vals == col2_vals).sum()
                approx_matches = ((col1_vals - col2_vals).abs() < 0.01).sum()
                max_diff = (col1_vals - col2_vals).abs().max() if len(matched) > 0 else None
                mean_diff = (col1_vals - col2_vals).abs().mean() if len(matched) > 0 else None
                
                value_comparisons[col] = {
                    'type': 'numeric',
                    'total_rows': len(matched),
                    'exact_matches': exact_matches,
                    'approx_matches': approx_matches,
                    'differences': len(matched) - exact_matches,
                    'max_difference': float(max_diff) if max_diff is not None else None,
                    'mean_difference': float(mean_diff) if mean_diff is not None else None,
                    'match_percentage': (exact_matches / len(matched) * 100) if len(matched) > 0 else 0
                }
            else:
                # String comparison
                exact_matches = (col1_vals == col2_vals).sum()
                null_both = (col1_vals.is_null() & col2_vals.is_null()).sum()
                null_one = ((col1_vals.is_null() & col2_vals.is_not_null()) | 
                           (col1_vals.is_not_null() & col2_vals.is_null())).sum()
                
                value_comparisons[col] = {
                    'type': 'string',
                    'total_rows': len(matched),
                    'exact_matches': exact_matches,
                    'null_both': null_both,
                    'null_one': null_one,
                    'differences': len(matched) - exact_matches - null_both,
                    'match_percentage': ((exact_matches + null_both) / len(matched) * 100) if len(matched) > 0 else 0
                }
    
    return {
        'total_rows_1': total_rows_1,
        'total_rows_2': total_rows_2,
        'matched_rows': matched_rows,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'match_percentage': (matched_rows / max(total_rows_1, total_rows_2) * 100) if max(total_rows_1, total_rows_2) > 0 else 0,
        'value_comparisons': value_comparisons
    }


def print_comparison_report(file1_path: str, file2_path: str, schema_comp: Dict, 
                           seq_comp: Dict, value_comp: Dict):
    """Print a formatted comparison report."""
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON REPORT")
    print("="*80)
    print(f"\nFile 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    # Schema comparison
    print("\n" + "-"*80)
    print("SCHEMA COMPARISON")
    print("-"*80)
    print(f"Field count - File 1: {schema_comp['field_count_1']}, File 2: {schema_comp['field_count_2']}")
    print(f"Field count match: {'[OK]' if schema_comp['field_count_match'] else '[FAIL]'}")
    print(f"Field names match: {'[OK]' if schema_comp['field_names_match'] else '[FAIL]'}")
    print(f"Field types match: {'[OK]' if schema_comp['field_types_match'] else '[FAIL]'}")
    
    if schema_comp['missing_in_2']:
        print(f"\n[WARNING] Fields in File 1 but not in File 2: {schema_comp['missing_in_2']}")
    if schema_comp['missing_in_1']:
        print(f"[WARNING] Fields in File 2 but not in File 1: {schema_comp['missing_in_1']}")
    if schema_comp['type_differences']:
        print(f"\n[WARNING] Type differences:")
        for col, diff in schema_comp['type_differences'].items():
            print(f"   {col}: {diff['type_1']} vs {diff['type_2']}")
    
    # Sequence comparison
    print("\n" + "-"*80)
    print("SEQUENCE COMPARISON")
    print("-"*80)
    print(f"Total sequences - File 1: {seq_comp['total_sequences_1']:,}, File 2: {seq_comp['total_sequences_2']:,}")
    print(f"Common sequences: {seq_comp['common_sequences']:,}")
    print(f"Sequence IDs match: {'[OK]' if seq_comp['sequence_ids_match'] else '[FAIL]'}")
    
    if seq_comp['only_in_1'] > 0:
        print(f"\n[WARNING] Sequences only in File 1: {seq_comp['only_in_1']}")
        if seq_comp['only_in_1_ids']:
            print(f"   Sample IDs: {seq_comp['only_in_1_ids']}")
    
    if seq_comp['only_in_2'] > 0:
        print(f"[WARNING] Sequences only in File 2: {seq_comp['only_in_2']}")
        if seq_comp['only_in_2_ids']:
            print(f"   Sample IDs: {seq_comp['only_in_2_ids']}")
    
    print(f"\nRow count matches: {seq_comp['row_count_matches']:,}/{seq_comp['common_sequences']:,}")
    if seq_comp['row_count_differences'] > 0:
        print(f"[WARNING] Sequences with different row counts: {seq_comp['row_count_differences']}")
        if seq_comp['row_count_differences_details']:
            print("   Sample differences:")
            for diff in seq_comp['row_count_differences_details'][:10]:
                print(f"      Sequence {diff['sequence_id']}: {diff['count_1']} vs {diff['count_2']} (diff: {diff['difference']:+d})")
    
    # Value comparison
    if 'error' not in value_comp:
        print("\n" + "-"*80)
        print("VALUE COMPARISON")
        print("-"*80)
        print(f"Total rows - File 1: {value_comp['total_rows_1']:,}, File 2: {value_comp['total_rows_2']:,}")
        print(f"Matched rows: {value_comp['matched_rows']:,}")
        print(f"Only in File 1: {value_comp['only_in_1']:,}")
        print(f"Only in File 2: {value_comp['only_in_2']:,}")
        print(f"Match percentage: {value_comp['match_percentage']:.2f}%")
        
        if value_comp['value_comparisons']:
            print(f"\nField-by-field comparison:")
            for col, comp in value_comp['value_comparisons'].items():
                print(f"\n  {col} ({comp['type']}):")
                print(f"    Total rows compared: {comp['total_rows']:,}")
                if comp['type'] == 'numeric':
                    print(f"    Exact matches: {comp['exact_matches']:,} ({comp['match_percentage']:.2f}%)")
                    print(f"    Approx matches (within 0.01): {comp['approx_matches']:,}")
                    print(f"    Differences: {comp['differences']:,}")
                    if comp['max_difference'] is not None:
                        print(f"    Max difference: {comp['max_difference']:.6f}")
                    if comp['mean_difference'] is not None:
                        print(f"    Mean difference: {comp['mean_difference']:.6f}")
                else:
                    print(f"    Exact matches: {comp['exact_matches']:,} ({comp['match_percentage']:.2f}%)")
                    print(f"    Both null: {comp.get('null_both', 0):,}")
                    print(f"    One null: {comp.get('null_one', 0):,}")
                    print(f"    Differences: {comp['differences']:,}")
    else:
        print(f"\n[WARNING] Value comparison error: {value_comp['error']}")
    
    print("\n" + "="*80)


@app.command()
def compare(
    file1: str = typer.Argument(..., help="Path to first checkpoint file"),
    file2: str = typer.Argument(..., help="Path to second checkpoint file"),
    key_columns: str = typer.Option(
        None,
        "--key-columns", "-k",
        help="Comma-separated list of key columns for row matching (default: sequence_id,Timestamp (YYYY-MM-DDThh:mm:ss))"
    )
):
    """
    Compare two checkpoint CSV files and provide detailed statistics.
    
    Examples:
        compare_checkpoints.py checkpoint1.csv checkpoint2.csv
        compare_checkpoints.py checkpoint1.csv checkpoint2.csv --key-columns "sequence_id,timestamp"
    """
    # Validate file paths
    file1_path = Path(file1)
    file2_path = Path(file2)
    
    if not file1_path.exists():
        typer.echo(f"[ERROR] File 1 does not exist: {file1}", err=True)
        raise typer.Exit(1)
    
    if not file2_path.exists():
        typer.echo(f"[ERROR] File 2 does not exist: {file2}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Loading files...")
    typer.echo(f"   File 1: {file1_path}")
    typer.echo(f"   File 2: {file2_path}")
    
    try:
        # Load DataFrames
        df1 = pl.read_csv(file1_path)
        df2 = pl.read_csv(file2_path)
        
        typer.echo(f"[OK] Loaded File 1: {len(df1):,} rows, {len(df1.columns)} columns")
        typer.echo(f"[OK] Loaded File 2: {len(df2):,} rows, {len(df2.columns)} columns")
        
        # Parse key columns if provided
        key_cols = None
        if key_columns:
            key_cols = [col.strip() for col in key_columns.split(',')]
            # Validate key columns exist
            missing_in_1 = [col for col in key_cols if col not in df1.columns]
            missing_in_2 = [col for col in key_cols if col not in df2.columns]
            if missing_in_1:
                typer.echo(f"[WARNING] Key columns not in File 1: {missing_in_1}", err=True)
            if missing_in_2:
                typer.echo(f"[WARNING] Key columns not in File 2: {missing_in_2}", err=True)
            key_cols = [col for col in key_cols if col in df1.columns and col in df2.columns]
        
        typer.echo(f"\nComparing schemas...")
        schema_comp = compare_schemas(df1, df2)
        
        typer.echo(f"Comparing sequences...")
        seq_comp = compare_sequences(df1, df2)
        
        typer.echo(f"Comparing values...")
        value_comp = compare_values(df1, df2, key_cols)
        
        # Print report
        print_comparison_report(str(file1_path), str(file2_path), schema_comp, seq_comp, value_comp)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        all_match = (
            schema_comp['field_count_match'] and
            schema_comp['field_names_match'] and
            schema_comp['field_types_match'] and
            seq_comp['sequence_ids_match'] and
            seq_comp['row_count_differences'] == 0 and
            value_comp.get('match_percentage', 0) == 100.0
        )
        
        if all_match:
            typer.echo("[OK] Files are identical!")
        else:
            typer.echo("[WARNING] Files differ - see details above")
            issues = []
            if not schema_comp['field_count_match'] or not schema_comp['field_names_match']:
                issues.append("Schema differences")
            if not seq_comp['sequence_ids_match']:
                issues.append("Sequence ID differences")
            if seq_comp['row_count_differences'] > 0:
                issues.append(f"{seq_comp['row_count_differences']} sequences with different row counts")
            if value_comp.get('match_percentage', 100) < 100:
                issues.append(f"Value differences ({100 - value_comp.get('match_percentage', 0):.2f}% mismatch)")
            typer.echo(f"   Issues found: {', '.join(issues)}")
        
    except Exception as e:
        typer.echo(f"[ERROR] Error during comparison: {e}", err=True)
        import traceback
        typer.echo(f"Traceback:\n{traceback.format_exc()}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

