#!/usr/bin/env python3
"""
University of Manchester (UoM) database converter.

This module provides the converter for UoM T1D databases (multi-user).
"""

from pathlib import Path
from typing import Optional

from formats.database_converters import MultiUserDatabaseConverter


class UoMDatabaseConverter(MultiUserDatabaseConverter):
    """Converter for University of Manchester T1D databases."""

    def _extract_user_id_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Extract user ID from UoM filename format.

        Args:
            file_path: Path to the CSV file

        Returns:
            User ID string or None if not found
        """
        filename = file_path.stem  # Get filename without extension

        # UoM format: UoMGlucose2301.csv -> participant ID: 2301
        # Also handles UoM2301sleeptime.csv -> participant ID: 2301
        if filename.startswith("UoM"):
            # Extract the ID part (digits)
            import re
            match = re.search(r"(\d+)", filename)
            if match:
                return match.group(1)

        return None

    def get_database_name(self) -> str:
        """Get the name of the database type."""
        return "University of Manchester T1D Database"


