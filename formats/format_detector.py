#!/usr/bin/env python3
"""
CSV format detector.

This module provides the format detection system that automatically identifies
CSV formats and returns the appropriate converter.
"""

from typing import Optional
from pathlib import Path
from .base_converter import CSVFormatConverter
from .dexcom_g6_converter import DexcomG6Converter
from .freestyle_libre3_converter import FreeStyleLibre3Converter
from .uom_glucose_converter import UoMGlucoseConverter
from .uom_basal_converter import UoMBasalConverter
from .uom_bolus_converter import UoMBolusConverter
from .uom_nutrition_converter import UoMNutritionConverter


class CSVFormatDetector:
    """Detects CSV format and returns appropriate converter."""
    
    def __init__(self):
        """Initialize the format detector with available converters."""
        self.converters = [
            DexcomG6Converter(),
            FreeStyleLibre3Converter(),
            UoMGlucoseConverter(),
            UoMBasalConverter(),
            UoMBolusConverter(),
            UoMNutritionConverter()
        ]
    
    def detect_format(self, file_path: Path) -> Optional[CSVFormatConverter]:
        """
        Detect the format of a CSV file and return appropriate converter.
        
        Args:
            file_path: Path to the CSV file to analyze
            
        Returns:
            Appropriate converter instance, or None if format is not supported
        """
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
                lines = file.readlines()
                if not lines:
                    return None
                
                # Try to find headers by checking multiple lines
                # Some formats have metadata lines before the actual headers
                for line_num in range(min(3, len(lines))):  # Check first 3 lines
                    line = lines[line_num].strip()
                    if not line:
                        continue
                    
                    # Parse CSV line properly to handle quoted headers
                    import csv
                    from io import StringIO
                    csv_reader = csv.reader(StringIO(line))
                    headers = next(csv_reader)
                    
                    # Clean headers: remove quotes and strip whitespace
                    headers = [col.strip().strip('"') for col in headers]
                    
                    # Check each converter
                    for converter in self.converters:
                        if converter.can_handle(headers):
                            # Set context for UoM converter if needed
                            if hasattr(converter, 'set_context'):
                                converter.set_context(file_path)
                            return converter
                
                return None
                
        except Exception as e:
            print(f"Error detecting format for {file_path}: {e}")
            return None
    
    def add_converter(self, converter: CSVFormatConverter) -> None:
        """
        Add a new converter to the detection system.
        
        Args:
            converter: New converter instance to add
        """
        self.converters.append(converter)
    
    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported format names.
        
        Returns:
            List of format names supported by this detector
        """
        return [converter.get_format_name() for converter in self.converters]
