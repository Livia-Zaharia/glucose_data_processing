"""
HUPA database converter.
"""

import re
from pathlib import Path
from typing import Optional

from formats.database_converters import MultiUserDatabaseConverter


class HupaDatabaseConverter(MultiUserDatabaseConverter):
    """Converter for HUPA database."""

    def _extract_user_id_from_filename(self, file_path: Path) -> Optional[str]:
        """
        Extract user ID from HUPA filename format.
        Example: HUPA0001P.csv -> 1p
        """
        stem = file_path.stem
        # Match HUPA followed by digits and then a trailing letter (P)
        match = re.search(r"HUPA(\d+)([a-zA-Z])", stem)
        if match:
            num = match.group(1).lstrip("0")
            if not num:  # Handle all zeros case like HUPA0000P
                num = "0"
            letter = match.group(2).lower()
            return f"{num}{letter}"
        return stem.lower()

    def get_database_name(self) -> str:
        return "HUPA Database"

