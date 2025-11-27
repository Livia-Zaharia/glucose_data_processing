#!/usr/bin/env python3
"""
Dexcom G6 format converter.

This module provides the converter for Dexcom G6 CSV format, which is the standard
format used by the glucose data preprocessor.
"""

from typing import List, Dict, Optional
from .base_converter import CSVFormatConverter


class DexcomG6Converter(CSVFormatConverter):
    """Converter for Dexcom G6 format (standard format)."""
    
    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.
        
        Args:
            headers: List of column headers from the CSV file
            
        Returns:
            True if this converter can handle the format, False otherwise
        """
        return 'Timestamp (YYYY-MM-DDThh:mm:ss)' in headers and 'Event Type' in headers
    
    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the standard format.
        
        Insulin is split into two columns based on Event Subtype:
        - Fast-Acting → Fast-Acting Insulin Value (u)
        - Long-Acting → Long-Acting Insulin Value (u)
        
        Args:
            row: Dictionary representing a single CSV row
            
        Returns:
            Dictionary in standard format, or None if row should be skipped
        """
        # Skip rows without timestamp
        timestamp = row.get('Timestamp (YYYY-MM-DDThh:mm:ss)', '').strip()
        if not timestamp:
            return None
        
        # Split insulin based on Event Subtype
        insulin_value = row.get('Insulin Value (u)', '').strip()
        event_subtype = row.get('Event Subtype', '').strip()
        
        fast_acting_insulin = ''
        long_acting_insulin = ''
        
        if insulin_value:
            if 'Fast-Acting' in event_subtype:
                fast_acting_insulin = insulin_value
            elif 'Long-Acting' in event_subtype:
                long_acting_insulin = insulin_value
            else:
                # If no subtype specified, default to fast-acting
                fast_acting_insulin = insulin_value
        
        return {
            'Timestamp (YYYY-MM-DDThh:mm:ss)': timestamp,
            'Event Type': row.get('Event Type', '').strip(),
            'Glucose Value (mg/dL)': row.get('Glucose Value (mg/dL)', '').strip(),
            'Fast-Acting Insulin Value (u)': fast_acting_insulin,
            'Long-Acting Insulin Value (u)': long_acting_insulin,
            'Carb Value (grams)': row.get('Carb Value (grams)', '').strip()
        }
    
    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.
        
        Returns:
            String name of the format
        """
        return "Dexcom G6"
