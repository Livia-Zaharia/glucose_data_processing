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
    
    # Default output fields (STANDARD NAMES).
    # These are fallback defaults if config is not provided.
    # NOTE: We intentionally use standard names so the pipeline can support arbitrary fields
    # without needing display-name mappings for every new field.
    DEFAULT_OUTPUT_FIELDS: List[str] = [
        'timestamp',
        'event_type',
        'glucose_value_mgdl',
        'fast_acting_insulin_u',
        'long_acting_insulin_u',
        'carb_grams',
    ]
    
    # Standard field name mapping (standard_name -> display_name)
    # These are fallback defaults if config is not provided
    STANDARD_FIELDS: Dict[str, str] = {
        "timestamp": "Timestamp (YYYY-MM-DDThh:mm:ss)",
        "event_type": "Event Type",
        "glucose_value_mgdl": "Glucose Value (mg/dL)",
        "fast_acting_insulin_u": "Fast-Acting Insulin Value (u)",
        "long_acting_insulin_u": "Long-Acting Insulin Value (u)",
        "carb_grams": "Carb Value (grams)"
    }
    
    # Class-level config-based field mappings (initialized from config)
    _config_default_output_fields: Optional[List[str]] = None
    _config_standard_fields: Optional[Dict[str, str]] = None
    
    _schema_cache: Optional[Dict] = None
    
    @classmethod
    def initialize_from_config(cls, config: Optional[Dict] = None) -> None:
        """
        Initialize field mappings from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'default_output_fields' and 'standard_fields' keys.
                   If None or keys missing, uses default hardcoded values.
        """
        if config is None:
            cls._config_default_output_fields = None
            cls._config_standard_fields = None
            return
        
        # Load default_output_fields from config
        if 'default_output_fields' in config and config['default_output_fields']:
            cls._config_default_output_fields = config['default_output_fields'].copy()
        else:
            cls._config_default_output_fields = None
        
        # Load standard_fields from config
        if 'standard_fields' in config and config['standard_fields']:
            cls._config_standard_fields = config['standard_fields'].copy()
        else:
            cls._config_standard_fields = None
    
    @classmethod
    def get_default_output_fields(cls) -> List[str]:
        """
        Get default output fields, using config if available, otherwise class defaults.
        
        Returns:
            List of default output field names (standard names)
        """
        if cls._config_default_output_fields is not None:
            return cls._config_default_output_fields.copy()
        return cls.DEFAULT_OUTPUT_FIELDS.copy()
    
    @classmethod
    def get_standard_fields(cls) -> Dict[str, str]:
        """
        Get standard field mappings, using config if available, otherwise class defaults.
        
        Returns:
            Dictionary mapping standard field names to display names
        """
        if cls._config_standard_fields is not None:
            return cls._config_standard_fields.copy()
        return cls.STANDARD_FIELDS.copy()
    
    def __init__(self, output_fields: Optional[List[str]] = None):
        """
        Initialize the converter with optional output fields configuration.
        
        Args:
            output_fields: List of standard field names to include in output. 
                          Uses standard field names (e.g., 'timestamp', 'glucose_value_mgdl').
                          If None, uses get_default_output_fields() (which uses config if available).
                          Timestamp is always included.
        """
        if output_fields is None:
            # Convert default display names to standard names
            # Use get_standard_fields() to get config-based or default mappings
            standard_fields = self.get_standard_fields()
            display_to_standard = {v: k for k, v in standard_fields.items()}
            default_output_fields = self.get_default_output_fields()
            self.output_fields_standard: Set[str] = {
                display_to_standard.get(f, f) for f in default_output_fields
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
        Return the standard field name as-is (for flexible field approach).
        
        Args:
            standard_name: Standard field name (e.g., 'timestamp', 'glucose_value_mgdl')
            
        Returns:
            The same standard field name (e.g., 'timestamp', 'glucose_value_mgdl')
        """
        # Return standard name directly - no conversion to display names
        # This supports flexible field approach where arbitrary fields can be added
        return standard_name
    
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
