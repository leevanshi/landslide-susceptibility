"""Pytest unit tests for spatial split isolation and prevention of block leakage."""
import numpy as np
import pytest

from common import load_data, spatial_split


@pytest.fixture(scope="module")
def dataset():
    return load_data()


def test_spatial_split_no_block_leakage(dataset):
    split, blocks = spatial_split(dataset, seed=42)

    train_blocks = set(blocks[split == 0])
    val_blocks = set(blocks[split == 1])
    test_blocks = set(blocks[split == 2])

    # STRICT ASSERTION: Intersection of blocks across splits must be entirely empty
    train_val_leakage = train_blocks & val_blocks
    train_test_leakage = train_blocks & test_blocks
    val_test_leakage = val_blocks & test_blocks

    assert len(train_val_leakage) == 0, f"Spatial leakage detected between train and val: {train_val_leakage}"
    assert len(train_test_leakage) == 0, f"Spatial leakage detected between train and test: {train_test_leakage}"
    assert len(val_test_leakage) == 0, f"Spatial leakage detected between val and test: {val_test_leakage}"


def test_spatial_split_fractions(dataset):
    split, _ = spatial_split(dataset, seed=42, fractions=(0.70, 0.15, 0.15))
    n = len(dataset)
    tr_ratio = (split == 0).sum() / n
    va_ratio = (split == 1).sum() / n
    te_ratio = (split == 2).sum() / n

    # Whole blocks are allocated, so tolerance within +/- 5%
    assert 0.65 <= tr_ratio <= 0.75, f"Train split out of bounds: {tr_ratio:.3f}"
    assert 0.10 <= va_ratio <= 0.20, f"Val split out of bounds: {va_ratio:.3f}"
    assert 0.10 <= te_ratio <= 0.20, f"Test split out of bounds: {te_ratio:.3f}"


def test_spatial_split_determinism(dataset):
    split1, blocks1 = spatial_split(dataset, seed=42)
    split2, blocks2 = spatial_split(dataset, seed=42)

    np.testing.assert_array_equal(split1, split2)
    np.testing.assert_array_equal(blocks1, blocks2)
