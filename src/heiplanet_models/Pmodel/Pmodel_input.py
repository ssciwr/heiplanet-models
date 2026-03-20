from dataclasses import dataclass

import xarray as xr


@dataclass
class PmodelInput:
    initial_conditions: xr.DataArray
    latitude: xr.DataArray
    population_density: xr.DataArray
    rainfall: xr.DataArray
    temperature: xr.DataArray
    temperature_mean: xr.DataArray

    def __repr__(self):
        attr_strings = []
        for attr, value in self.__dict__.items():
            type_name = type(value).__name__
            shape_str = ""
            if hasattr(value, "shape"):
                shape_str = f", shape={value.shape}"
            attr_strings.append(f"\n\t{attr}: {type_name}{shape_str}")
        attrs = ",".join(attr_strings)
        return f"{self.__class__.__name__}({attrs})"
