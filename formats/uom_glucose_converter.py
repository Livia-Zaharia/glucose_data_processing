#!/usr/bin/env python3
"""
University of Manchester (UoM) glucose data format converter.

This module provides the converter for UoM glucose data format.
"""

from typing import List, Dict, Optional
from datetime import datetime
from .base_converter import CSVFormatConverter


class UoMGlucoseConverter(CSVFormatConverter):
    """Converter for University of Manchester glucose data format."""
    
    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.
        
        Args:
            headers: List of column headers from the CSV file
            
        Returns:
            True if this converter can handle the format, False otherwise
        """
        # Clean headers (remove empty strings and BOM characters)
        clean_headers = [h.strip().lstrip('\ufeff') for h in headers if h.strip()]
        
        # Check for UoM glucose format headers
        uom_glucose_headers = ['bg_ts', 'value']
        
        return all(header in clean_headers for header in uom_glucose_headers)
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """
        Parse UoM timestamp format to standard format.
        
        Args:
            timestamp_str: Timestamp string in UoM format (MM/DD/YYYY HH:MM or MM/DD/YYYY HH:MM:SS)
            
        Returns:
            Timestamp string in standard format (YYYY-MM-DDTHH:MM:SS) or None if parsing fails
        """
        if not timestamp_str or timestamp_str.strip() == "":
            return None
        
        timestamp_str = timestamp_str.strip()
        
        # Try different UoM timestamp formats
        formats = [
            "%m/%d/%Y %H:%M:%S",  # MM/DD/YYYY HH:MM:SS
            "%m/%d/%Y %H:%M",     # MM/DD/YYYY HH:MM
            "%d/%m/%Y %H:%M:%S",  # DD/MM/YYYY HH:MM:SS
            "%d/%m/%Y %H:%M",     # DD/MM/YYYY HH:MM
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.strftime('%Y-%m-%dT%H:%M:%S')
            except ValueError:
                continue
        
        return None
    
    def _convert_glucose_value(self, value_str: str) -> Optional[str]:
        """
        Convert glucose value from mmol/L to mg/dL.
        
        Args:
            value_str: Glucose value string in mmol/L
            
        Returns:
            Glucose value string in mg/dL or None if conversion fails
        """
        if not value_str or value_str.strip() == "":
            return None
        
        try:
            value_mmol = float(value_str.strip())
            value_mgdl = value_mmol * 18  # Convert mmol/L to mg/dL
            return str(round(value_mgdl, 1))
        except (ValueError, TypeError):
            return None
    
    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the standard format.
        
        Args:
            row: Dictionary representing a single CSV row
            
        Returns:
            Dictionary in standard format, or None if row should be skipped
        """
        # Initialize result with standard fields
        result = {
            'Timestamp (YYYY-MM-DDThh:mm:ss)': '',
            'Event Type': '',
            'Glucose Value (mg/dL)': '',
            'Fast-Acting Insulin Value (u)': '',
            'Long-Acting Insulin Value (u)': '',
            'Carb Value (grams)': ''
        }
        
        # Helper function to get value with BOM handling
        def get_clean_value(key):
            value = row.get(key, '')
            if not value:
                # Try with BOM prefix
                bom_key = '\ufeff' + key
                value = row.get(bom_key, '')
            return value
        
        # Convert glucose data
        timestamp = self._parse_timestamp(get_clean_value('bg_ts'))
        if not timestamp:
            return None
        
        glucose_value = self._convert_glucose_value(get_clean_value('value'))
        if glucose_value is None:
            return None
        
        result['Timestamp (YYYY-MM-DDThh:mm:ss)'] = timestamp
        result['Event Type'] = 'EGV'  # Estimated Glucose Value
        result['Glucose Value (mg/dL)'] = glucose_value
        
        return result
    
    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.
        
        Returns:
            String name of the format
        """
        return "UoM Glucose Data"
