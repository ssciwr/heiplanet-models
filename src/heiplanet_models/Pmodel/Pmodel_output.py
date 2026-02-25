"""Output utilities for allocating P-model simulation arrays."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PmodelOutput:
    """Container for model output arrays.

    Attributes:
        model_output (np.ndarray): Array with shape
            ``(longitude, latitude, ode_variable, time)``.
    """

    model_output: np.ndarray

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
