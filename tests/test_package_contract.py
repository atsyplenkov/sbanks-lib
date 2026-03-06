# -*- coding: utf-8 -*-
"""Package contract tests for the renamed sbanks distribution."""

import importlib
import tomllib
from pathlib import Path


def test_public_namespace_importable_as_sbanks():
    module = importlib.import_module("sbanks")
    assert module is not None


def test_expected_public_symbols_available_from_sbanks():
    module = importlib.import_module("sbanks")
    for name in [
        "WhittakerSmoother",
        "smooth_open_geometry",
        "smooth_closed_geometry",
        "resample_and_smooth",
    ]:
        assert hasattr(module, name)


def test_distribution_metadata_uses_sbanks_name():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    assert data["project"]["name"] == "sbanks"
