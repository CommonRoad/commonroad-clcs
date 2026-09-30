# standard imports
from typing import Optional
from abc import abstractmethod, ABCMeta
from copy import deepcopy

# third party
import numpy as np

from commonroad_clcs.config import (
    CLCSParams
)


class IReferencePathProcessor(metaclass=ABCMeta):
    """
    Abstract Interface class for different pre-processors of the reference path.
    """
    def __init__(self, params: CLCSParams):
        """
        :param params: parameter from configuration
        """
        self._params: CLCSParams = params
        self._ref_path_original: Optional[np.ndarray] = None

    def __call__(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        """Callable interface: Processes a reference path with the given option."""
        self._ref_path_original = deepcopy(ref_path_input)
        return self._process_path(deepcopy(ref_path_input), *args, **kwargs)

    @abstractmethod
    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        """
        Processes a reference path with the given option. Returns the pre-processed reference path.
        Abstract method, which is implemented in derived processing strategy classes.
        """
        pass

    @property
    def ref_path_original(self) -> Optional[np.ndarray]:
        """Getter for original (input) reference path"""
        return self._ref_path_original
