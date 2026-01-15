"""Configuration utilities for glucose data processing."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from processing.core.fields import StandardFieldNames

def extract_field_categories(database_type: str) -> Dict[str, Any]:
    """
    Extract field categories and settings from schema file.
    
    Args:
        database_type: Database type (e.g., 'uom', 'dexcom', 'freestyle_libre3')
        
    Returns:
        Dictionary with categories ('continuous', 'occasional', 'service') 
        and settings (e.g., 'remove_after_calibration')
    """
    # Map database type to schema file name (legacy aliases).
    # Prefer convention: `<database_type>_schema.yaml` if present.
    schema_files = {
        'uom': 'uom_schema.yaml',
        'dexcom': 'dexcom_schema.yaml',
        'libre3': 'freestyle_libre3_schema.yaml',
        'freestyle_libre3': 'freestyle_libre3_schema.yaml',
        'ai_ready': 'ai_ready_schema.yaml',
        'hupa': 'hupa_schema.yaml',
    }

    schema_file = schema_files.get(database_type, f"{database_type}_schema.yaml")
    
    # Load schema file
    # Note: Using Path(__file__).parent.parent.parent to get to the root from processing/core/
    root_dir = Path(__file__).parent.parent.parent
    schema_path = root_dir / 'formats' / schema_file
    
    if not schema_path.exists():
        # Return default with only glucose as continuous
        return {
            'continuous': [StandardFieldNames.GLUCOSE_VALUE],
            'occasional': [],
            'service': [],
            'remove_after_calibration': True
        }
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    
    # Get field_categories from schema
    field_categories = schema.get('field_categories', {})
    
    # Build result dictionary using standard field names directly
    result = {
        'continuous': [],
        'occasional': [],
        'service': [],
        'remove_after_calibration': schema.get('remove_after_calibration', True)
    }
    
    for standard_name, category in field_categories.items():
        if category in result:
            result[category].append(standard_name)
    
    # Always ensure glucose is in continuous (if it exists)
    glucose_col = StandardFieldNames.GLUCOSE_VALUE
    if glucose_col not in result['continuous']:
        result['continuous'].append(glucose_col)
    
    return result

