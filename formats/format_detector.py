#!/usr/bin/env python3
"""
CSV format detector.

This module provides the format detection system that automatically identifies
CSV formats and returns the appropriate converter.
"""

from pathlib import Path
from typing import List, Optional
from loguru import logger
from formats.base_converter import CSVFormatConverter
from formats.dexcom.dexcom_g6_converter import DexcomG6Converter
from formats.libre3.freestyle_libre3_converter import FreeStyleLibre3Converter
from formats.uom.uom_activity_converter import UoMActivityConverter
from formats.uom.uom_basal_converter import UoMBasalConverter
from formats.uom.uom_bolus_converter import UoMBolusConverter
from formats.uom.uom_glucose_converter import UoMGlucoseConverter
from formats.uom.uom_nutrition_converter import UoMNutritionConverter
from formats.uom.uom_sleep_converter import UoMSleepConverter
from formats.uom.uom_sleeptime_converter import UoMSleeptimeConverter
from formats.hupa.hupa_converter import HupaConverter
from formats.medtronic.medtronic_converter import MedtronicConverter
from formats.minidose1.minidose1_converter import Minidose1Converter


class CSVFormatDetector:
    """Detects CSV format and returns appropriate converter."""
    
    def __init__(self, output_fields: Optional[List[str]] = None):
        """
        Initialize the format detector.
        
        Args:
            output_fields: List of field names to include in converter output.
                          If None, uses default fields.
        """
        self.output_fields = output_fields
    
    def _get_converters(self) -> List[CSVFormatConverter]:
        """Get list of converter instances with configured output_fields."""
        converters = [
            DexcomG6Converter(self.output_fields),
            FreeStyleLibre3Converter(self.output_fields),
            UoMGlucoseConverter(self.output_fields),
            UoMBasalConverter(self.output_fields),
            UoMBolusConverter(self.output_fields),
            UoMNutritionConverter(self.output_fields),
            UoMActivityConverter(self.output_fields),
            UoMSleepConverter(self.output_fields),
            UoMSleeptimeConverter(self.output_fields),
            HupaConverter(self.output_fields),
            MedtronicConverter(self.output_fields),
            Minidose1Converter(self.output_fields)
        ]
        return converters
    
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
                for line_num in range(min(15, len(lines))):  # Check first 15 lines (Medtronic has 7-8 lines of header)
                    line = lines[line_num].strip()
                    if not line:
                        continue
                    
                    # Parse CSV line properly to handle different delimiters
                    import csv
                    from io import StringIO
                    
                    # Try common delimiters
                    possible_delimiters = [',', ';', '|', '\t']
                    best_headers = []
                    
                    for delim in possible_delimiters:
                        csv_reader = csv.reader(StringIO(line), delimiter=delim)
                        try:
                            headers = next(csv_reader)
                            # Clean headers: remove quotes and strip whitespace
                            headers = [col.strip().strip('"') for col in headers]
                            
                            if len(headers) > len(best_headers):
                                best_headers = headers
                                
                            # Check each converter with these headers
                            for converter in self._get_converters():
                                if converter.can_handle(headers):
                                    # Set context for converter if needed
                                    if hasattr(converter, 'set_context'):
                                        converter.set_context(file_path)
                                    return converter
                        except Exception:
                            continue
                
                return None
                
        except Exception as e:
            logger.info(f"Error detecting format for {file_path}: {e}")
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
        return [converter.get_format_name() for converter in self._get_converters()]
