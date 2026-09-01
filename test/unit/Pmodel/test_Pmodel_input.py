"""Unit tests for heiplanet_models.Pmodel.Pmodel_input.

Tests are organized by function with clear visual separation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pytest
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_input import PmodelInput


# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def dummy_pmodel_input():
    """Fixture for a fully populated PmodelInput instance."""
    initial_conditions = np.zeros((2, 2, 3))
    latitude = xr.DataArray(np.linspace(-10, 10, 2), dims="latitude")
    population_density = xr.DataArray(np.ones((2, 2)), dims=("longitude", "latitude"))
    rainfall = xr.DataArray(np.zeros((2, 2)), dims=("longitude", "latitude"))
    temperature = xr.DataArray(
        np.full((2, 2, 3), 25.0), dims=("longitude", "latitude", "time")
    )
    temperature_mean = xr.DataArray(
        np.full((2, 2, 3), 20.0), dims=("longitude", "latitude", "time")
    )
    return PmodelInput(
        initial_conditions=initial_conditions,
        latitude=latitude,
        population_density=population_density,
        rainfall=rainfall,
        temperature=temperature,
        temperature_mean=temperature_mean,
    )


# =============================================================================
# TESTS: PmodelInput
# =============================================================================


class TestPmodelInput:
    """Test suite for PmodelInput."""

    def test_instantiation_and_types(self, dummy_pmodel_input):
        assert isinstance(dummy_pmodel_input, PmodelInput)
        assert isinstance(dummy_pmodel_input.initial_conditions, np.ndarray)
        assert isinstance(dummy_pmodel_input.latitude, xr.DataArray)
        assert isinstance(dummy_pmodel_input.population_density, xr.DataArray)
        assert isinstance(dummy_pmodel_input.rainfall, xr.DataArray)
        assert isinstance(dummy_pmodel_input.temperature, xr.DataArray)
        assert isinstance(dummy_pmodel_input.temperature_mean, xr.DataArray)

    def test_repr_contains_class_and_attrs(self, dummy_pmodel_input):
        rep = repr(dummy_pmodel_input)
        assert "PmodelInput" in rep
        for attr in [
            "initial_conditions",
            "latitude",
            "population_density",
            "rainfall",
            "temperature",
            "temperature_mean",
        ]:
            assert attr in rep

    def test_missing_required_attributes(self):
        with pytest.raises(TypeError):
            PmodelInput()
