# standard imports
from typing import Dict, Callable, Optional

# third party
import numpy as np

from scipy.interpolate import (
    interp1d,
    CubicSpline,
    Akima1DInterpolator
)


class Interpolator:
    """
    Factory class for creating 1d interpolation functions.

    Supported interpolation types:
    - "linear": Piecewise linear interpolation
    - "cubic": Cubic spline interpolation
    - "akima": Akima spline interpolation
    """
    _dict_interp_functions: Dict[str, Callable] = {
        "linear": interp1d,
        "cubic": CubicSpline,
        "akima": Akima1DInterpolator
    }

    @classmethod
    def get_function(cls,
                     x: np.ndarray,
                     y: np.ndarray,
                     interp_type: str = "linear",
                     **kwargs):
        """
        Returns an interpolation function for the given interpolation type
        :param x: 1d array of independent variable
        :param y: 1d array of dependent variable
        :param interp_type: string for the type of interpolation
        """
        func: Optional[Callable] = cls._dict_interp_functions.get(interp_type)

        if func is None:
            raise KeyError(f"Unsupported interpolation type: {interp_type}. "
                           f"Supported types: {list(cls._dict_interp_functions.keys())}")
        return func(x, y, **kwargs)