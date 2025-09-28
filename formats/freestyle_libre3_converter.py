#!/usr/bin/env python3
"""
FreeStyle Libre 3 format converter.

This module provides the converter for FreeStyle Libre 3 CSV format.
"""

from typing import List, Dict, Optional
from datetime import datetime
from .base_converter import CSVFormatConverter


class FreeStyleLibre3Converter(CSVFormatConverter):
    """Converter for FreeStyle Libre 3 format."""
    
    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.
        
        Args:
            headers: List of column headers from the CSV file
            
        Returns:
            True if this converter can handle the format, False otherwise
        """
        return 'Device Timestamp' in headers and 'Historic Glucose mg/dL' in headers
    
    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the standard format.
        
        Args:
            row: Dictionary representing a single CSV row
            
        Returns:
            Dictionary in standard format, or None if row should be skipped
        """
        # Skip rows without timestamp
        device_timestamp = row.get('Device Timestamp', '').strip()
        if not device_timestamp:
            return None
        
        # Convert timestamp format from "DD-MM-YYYY HH:MM" to "YYYY-MM-DDTHH:MM:SS"
        try:
            dt = datetime.strptime(device_timestamp, "%d-%m-%Y %H:%M")
            formatted_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            # Try alternative format if the first one fails
            try:
                dt = datetime.strptime(device_timestamp, "%d-%m-%Y %H:%M:%S")
                formatted_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                print(f"Warning: Could not parse timestamp '{device_timestamp}', skipping row")
                return None
        
        # Determine event type based on record type
        record_type = row.get('Record Type', '').strip()
        if record_type == '0':
            event_type = 'EGV'  # Estimated Glucose Value
        elif record_type == '6':
            event_type = 'Note'  # Manual entry/note
        else:
            event_type = 'Unknown'
        
        # Get glucose value (prefer historic over scan)
        glucose_value = row.get('Historic Glucose mg/dL', '').strip()
        if not glucose_value:
            glucose_value = row.get('Scan Glucose mg/dL', '').strip()
        
        # Combine insulin values (rapid-acting + meal + correction)
        rapid_insulin = self._safe_float(row.get('Rapid-Acting Insulin (units)', ''))
        meal_insulin = self._safe_float(row.get('Meal Insulin (units)', ''))
        correction_insulin = self._safe_float(row.get('Correction Insulin (units)', ''))
        total_insulin = rapid_insulin + meal_insulin + correction_insulin
        
        # Get carb value
        carb_value = row.get('Carbohydrates (grams)', '').strip()
        
        return {
            'Timestamp (YYYY-MM-DDThh:mm:ss)': formatted_timestamp,
            'Event Type': event_type,
            'Glucose Value (mg/dL)': glucose_value,
            'Insulin Value (u)': str(total_insulin) if total_insulin > 0 else '',
            'Carb Value (grams)': carb_value
        }
    
    def _safe_float(self, value: str) -> float:
        """
        Safely convert string to float, returning 0 if conversion fails.
        
        Args:
            value: String value to convert
            
        Returns:
            Float value or 0.0 if conversion fails
        """
        try:
            return float(value.strip()) if value.strip() else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.
        
        Returns:
            String name of the format
        """
        return "FreeStyle Libre 3"
