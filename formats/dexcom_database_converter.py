#!/usr/bin/env python3
"""
Dexcom database converter.

This module provides the converter for Dexcom G6 databases (mono-user).
Handles High/Low value replacement and calibration removal specific to Dexcom.
"""

from typing import Dict, Any
import polars as pl
from .database_converters import MonoUserDatabaseConverter


class DexcomDatabaseConverter(MonoUserDatabaseConverter):
    """Converter for Dexcom G6 databases."""
    
    def _apply_database_specific_processing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Dexcom-specific processing: High/Low replacement and calibration removal.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        # Get Dexcom-specific config
        dexcom_config = self.config.get('dexcom', {})
        high_value = dexcom_config.get('high_glucose_value', 401)
        low_value = dexcom_config.get('low_glucose_value', 39)
        remove_calibration = dexcom_config.get('remove_calibration', True)
        
        # Step 1: Replace High/Low glucose values
        if 'Glucose Value (mg/dL)' in df.columns:
            print("Applying Dexcom-specific High/Low value replacement...")
            
            # Count High and Low values before replacement
            high_count = df.filter(pl.col('Glucose Value (mg/dL)') == 'High').height
            low_count = df.filter(pl.col('Glucose Value (mg/dL)') == 'Low').height
            
            if high_count > 0 or low_count > 0:
                print(f"  Replacing {high_count} 'High' values with {high_value}")
                print(f"  Replacing {low_count} 'Low' values with {low_value}")
                
                # Replace High and Low with configurable values, then convert to float
                df = df.with_columns([
                    pl.col('Glucose Value (mg/dL)')
                    .str.replace('High', str(high_value))
                    .str.replace('Low', str(low_value))
                    .cast(pl.Float64, strict=False)
                    .alias('Glucose Value (mg/dL)')
                ])
                print("  OK: Glucose field converted to Float64 type")
        
        # Step 2: Remove calibration events
        if remove_calibration and 'Event Type' in df.columns:
            print("Applying Dexcom-specific calibration event removal...")
            
            calibration_events = df.filter(pl.col('Event Type') == 'Calibration')
            calibration_count = len(calibration_events)
            
            if calibration_count > 0:
                print(f"  Removing {calibration_count} calibration events")
                df = df.filter(pl.col('Event Type') != 'Calibration')
                print("  OK: Calibration events removed")
            else:
                print("  No calibration events found")
        
        return df
    
    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "Dexcom G6 Database"
