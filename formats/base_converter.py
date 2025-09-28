#!/usr/bin/env python3
"""
Base converter class for CSV format converters.

This module provides the abstract base class that all format converters must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class CSVFormatConverter(ABC):
    """Abstract base class for CSV format converters."""
    
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
        Convert a single row to the standard format.
        
        Args:
            row: Dictionary representing a single CSV row
            
        Returns:
            Dictionary in standard format, or None if row should be skipped
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
