#!/usr/bin/env python3
"""
Database detector for identifying database types.

This module provides the database detection system that automatically identifies
database types and returns the appropriate database converter.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from .database_converters import DatabaseConverter
from .dexcom_database_converter import DexcomDatabaseConverter
from .libre3_database_converter import Libre3DatabaseConverter
from .uom_database_converter import UoMDatabaseConverter


class DatabaseDetector:
    """Detects database type and returns appropriate database converter."""
    
    def __init__(self):
        """Initialize the database detector with available converters."""
        self.database_converters = {
            'dexcom': DexcomDatabaseConverter,
            'libre3': Libre3DatabaseConverter,
            'uom': UoMDatabaseConverter
        }
    
    def detect_database_type(self, data_folder: str) -> str:
        """
        Detect the database type from the folder structure and file patterns.
        
        Args:
            data_folder: Path to the data folder to analyze
            
        Returns:
            Database type string ('dexcom', 'libre3', 'uom', or 'unknown')
        """
        data_path = Path(data_folder)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        # Get all CSV files
        csv_files = list(data_path.glob("**/*.csv"))
        
        if not csv_files:
            return 'unknown'
        
        # Analyze file patterns to determine database type
        file_patterns = {
            'dexcom': 0,
            'libre3': 0,
            'uom': 0
        }
        
        for csv_file in csv_files:
            filename = csv_file.stem.lower()
            
            # Check for Dexcom patterns (standard format files)
            if any(pattern in filename for pattern in ['dexcom', 'g6', 'cgm']):
                file_patterns['dexcom'] += 1
            elif any(pattern in filename for pattern in ['libre', 'freestyle']):
                file_patterns['libre3'] += 1
            elif filename.startswith('uom'):
                file_patterns['uom'] += 1
            else:
                # Check file content to determine format
                try:
                    with open(csv_file, 'r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
                        first_lines = [file.readline().strip() for _ in range(3)]
                        
                        # Check for Dexcom format headers
                        for line in first_lines:
                            if 'Timestamp (YYYY-MM-DDThh:mm:ss)' in line and 'Event Type' in line:
                                file_patterns['dexcom'] += 1
                                break
                        
                        # Check for Libre3 format headers
                        for line in first_lines:
                            if 'Device Timestamp' in line and 'Historic Glucose mg/dL' in line:
                                file_patterns['libre3'] += 1
                                break
                        
                        # Check for UoM format headers
                        for line in first_lines:
                            if any(header in line for header in ['bg_ts', 'basal_ts', 'bolus_ts', 'meal_ts']):
                                file_patterns['uom'] += 1
                                break
                                
                except Exception:
                    # If we can't read the file, skip it
                    continue
        
        # Determine the most likely database type
        if max(file_patterns.values()) == 0:
            return 'unknown'
        
        return max(file_patterns, key=file_patterns.get)
    
    def get_database_converter(self, database_type: str, config: Dict[str, Any]) -> Optional[DatabaseConverter]:
        """
        Get the appropriate database converter for the given database type.
        
        Args:
            database_type: Database type string
            config: Configuration dictionary
            
        Returns:
            Database converter instance or None if type not supported
        """
        if database_type in self.database_converters:
            return self.database_converters[database_type](config)
        else:
            return None
    
    def get_supported_database_types(self) -> list[str]:
        """
        Get list of supported database types.
        
        Returns:
            List of database type names supported by this detector
        """
        return list(self.database_converters.keys())
