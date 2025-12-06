#!/usr/bin/env python3
"""
Base converter class for CSV format converters.

This module provides the abstract base class that all format converters must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
import json
from pathlib import Path


class CSVFormatConverter(ABC):
    """Abstract base class for CSV format converters."""
    
    # Default output fields matching current glucose_ml_preprocessor.py usage
    DEFAULT_OUTPUT_FIELDS: List[str] = [
        'Timestamp (YYYY-MM-DDThh:mm:ss)',
        'Event Type',
        'Glucose Value (mg/dL)',
        'Fast-Acting Insulin Value (u)',
        'Long-Acting Insulin Value (u)',
        'Carb Value (grams)'
    ]
    
    # Standard field name mapping (standard_name -> display_name)
    STANDARD_FIELDS: Dict[str, str] = {
        "timestamp": "Timestamp (YYYY-MM-DDThh:mm:ss)",
        "event_type": "Event Type",
        "glucose_value_mgdl": "Glucose Value (mg/dL)",
        "fast_acting_insulin_u": "Fast-Acting Insulin Value (u)",
        "long_acting_insulin_u": "Long-Acting Insulin Value (u)",
        "carb_grams": "Carb Value (grams)"
    }
    
    _schema_cache: Optional[Dict] = None
    
    def __init__(self, output_fields: Optional[List[str]] = None):
        """
        Initialize the converter with optional output fields configuration.
        
        Args:
            output_fields: List of standard field names to include in output. 
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses DEFAULT_OUTPUT_FIELDS.
                          Timestamp is always included.
        """
        if output_fields is None:
            # Convert default display names to standard names
            display_to_standard = {v: k for k, v in self.STANDARD_FIELDS.items()}
            self.output_fields_standard: Set[str] = {
                display_to_standard.get(f, f) for f in self.DEFAULT_OUTPUT_FIELDS
            }
        else:
            # Use provided standard field names
            self.output_fields_standard = set(output_fields)
        
        # Always include timestamp
        self.output_fields_standard.add('timestamp')
    
    @classmethod
    def _load_schema(cls, schema_file: str) -> Dict:
        """
        Load conversion schema from JSON file.
        
        Args:
            schema_file: Name of the schema file (e.g., 'uom_schema.json')
            
        Returns:
            Dictionary containing conversion schema
        """
        schema_path = Path(__file__).parent / schema_file
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_standard_field_name(self, standard_name: str) -> str:
        """
        Convert standard field name to display name.
        
        Args:
            standard_name: Standard field name (e.g., 'timestamp', 'glucose_value_mgdl')
            
        Returns:
            Display field name (e.g., 'Timestamp (YYYY-MM-DDThh:mm:ss)')
        """
        return self.STANDARD_FIELDS.get(standard_name, standard_name)
    
    def _filter_output(self, result: Dict[str, str]) -> Dict[str, str]:
        """
        Filter output dictionary to only include requested fields.
        
        Args:
            result: Dictionary with standard field names as keys
            
        Returns:
            Dictionary with display field names, filtered to requested fields
        """
        filtered = {}
        for standard_name, value in result.items():
            if standard_name in self.output_fields_standard:
                display_name = self._get_standard_field_name(standard_name)
                filtered[display_name] = value
        return filtered
    
    def _get_clean_value(self, row: Dict[str, str], key: str) -> str:
        """
        Get value from row with BOM handling.
        
        Args:
            row: Dictionary representing a CSV row
            key: Field name to retrieve
            
        Returns:
            Cleaned value string
        """
        value = row.get(key, '')
        if not value:
            # Try with BOM prefix
            bom_key = '\ufeff' + key
            value = row.get(bom_key, '')
        return value.strip() if value else ''
    
    def _get_source_fields(self, row: Dict[str, str]) -> set:
        """
        Get all source field names from row, normalizing BOM keys.
        
        Args:
            row: Dictionary representing a CSV row
            
        Returns:
            Set of source field names
        """
        source_fields = set()
        for key in row.keys():
            if key.startswith('\ufeff'):
                source_fields.add(key[1:])  # Remove BOM
            else:
                source_fields.add(key)
        return source_fields
    
    @abstractmethod
    def can_handle(self, headers: List[str]) -> bool:
        """
        Check if this converter can handle the given CSV headers.
        
        Args:
            headers: List of column headers from the CSV file
            
        Returns:
            True if this converter can handle the format, False otherwise
        """
        pass
    
    @abstractmethod
    def convert_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        Convert a single row to the configured output format.
        
        Args:
            row: Dictionary representing a single CSV row
            
        Returns:
            Dictionary with requested fields, or None if row should be skipped
        """
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """
        Get the name of the format this converter handles.
        
        Returns:
            String name of the format
        """
        pass
