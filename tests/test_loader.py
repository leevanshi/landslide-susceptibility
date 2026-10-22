"""Pytest unit tests for Alvioli benchmark loader, schema validation, and synthetic fallback."""
import pandas as pd
import pytest

from common import (SchemaValidationError, load_data, map_alvioli_columns,
                    validate_dataset_schema)


def test_schema_validation_valid():
    df = load_data()
    is_valid, errors = validate_dataset_schema(df)
    assert is_valid
    assert len(errors) == 0


def test_schema_validation_missing_columns():
    invalid_df = pd.DataFrame({
        "su_id": ["SU-01"],
        "slope": [25.0],
    })
    is_valid, errors = validate_dataset_schema(invalid_df)
    assert not is_valid
    assert any("Missing required columns" in e for e in errors)


def test_alvioli_column_mapping():
    raw_benchmark_row = pd.DataFrame({
        "su_id": ["SU-1001"],
        "x_coord": [12.4],
        "y_coord": [42.8],
        "slope_mean": [28.2],
        "elevation_mean": [520.0],
        "aspect_mean": [145.0],
        "curv_mean": [0.04],
        "precip_mean": [980.0],
        "dist_river": [320.0],
        "dist_road": [210.0],
        "dist_fault": [450.0],
        "landcover": ["forest"],
        "litho": ["sandstone"],
        "landslide_presence": [1],
    })
    mapped = map_alvioli_columns(raw_benchmark_row)
    is_valid, errors = validate_dataset_schema(mapped)
    assert is_valid, f"Mapped columns failed validation: {errors}"
    assert "unit_id" in mapped.columns
    assert "slope" in mapped.columns
    assert "rainfall" in mapped.columns
    assert "landslide" in mapped.columns


def test_synthetic_fallback():
    # Loading default data without benchmark present falls back cleanly to synthetic
    df = load_data()
    assert df.attrs.get("is_synthetic") is True
    assert "Synthetic Simulation" in df.attrs.get("provenance", "")
