from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from commonroad.common.validity import (
    ValidTypes,
    is_valid_orientation
)

import commonroad_clcs.pycrccosy as pycrccosy

from commonroad_clcs.config import CLCSParams
from commonroad_clcs.util import (
    compute_pathlength_from_polyline,
    compute_orientation_from_polyline,
    remove_duplicated_points_from_polyline, compute_curvature_from_polyline_python
)
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory


class RefPathLengthException(Exception):
    pass


# threshold for checking orientation jumps in reference path
_ANGLE_THRESHOLD = np.pi / 2


class CurvilinearCoordinateSystem(pycrccosy.CurvilinearCoordinateSystem):
    """
    Python wrapper class for pycrccosy.CurvilinearCoordinateSystem.
    This class takes a reference path and a CLCS configuration and
        - pre-processes the reference path based on user options (e.g., smoothing, curvature reduction etc...)
        - instantiates the C++ CurvilinearCoordinateSystem object
        - allows interacting with the C++ object via the pybind methods
    """

    def __init__(
            self,
            reference_path: np.ndarray,
            params: CLCSParams,
            preprocess_path=True,
            validity_checks=True,
    ):
        """
        :param reference_path: reference path as numpy ndarray
        :param params: config parameters for reference path pre-processing and CLCS
        :param preprocess_path: Flag indicating whether the reference path should be pre-processed
                                Set to False if reference path is pre-processed before already
        :param validity_checks: If True, performs validity checks for the reference path before instantiation
        """
        # reference path checks
        if validity_checks:
            self.check_ref_path_validity(reference_path)

        if preprocess_path:
            ref_path_processor = ProcessorFactory.create_processor(params)
            reference_path = ref_path_processor(reference_path)

        # remove potential duplicate vertices
        reference_path = remove_duplicated_points_from_polyline(reference_path)

        # init base class (C++ backend)
        super().__init__(
            reference_path,
            params.default_proj_domain_limit,
            params.eps,
            params.eps2,
            log_level=params.logging_level,
            method=params.method,
        )

        # pre-compute reference path attributes
        self._reference_path = np.asarray(super().reference_path())
        self._ref_pos = compute_pathlength_from_polyline(self.ref_path)
        self._ref_theta = np.unwrap(compute_orientation_from_polyline(self.ref_path))
        self._ref_curv = compute_curvature_from_polyline_python(self.ref_path)
        self._ref_curv_d = np.gradient(self._ref_curv, self.ref_pos)
        super().set_curvature(self.ref_curv)

    @staticmethod
    def check_ref_path_validity(ref_path: np.ndarray):
        """Validity checks for input reference path for CLCS"""
        # check valid type
        assert isinstance(ref_path, ValidTypes.ARRAY), "Reference path needs to be a numpy array"
        # check number of points in ref path
        assert len(ref_path) >= 3, "Reference path has to contain >= 3 points"
        # check valid orientation values
        theta_arr = np.unwrap(compute_orientation_from_polyline(ref_path))
        assert all(is_valid_orientation(theta) for theta in theta_arr), \
            "Reference path orientations should be in [-2pi, 2pi]"
        # check orientation jumps
        diff_theta_arr = np.diff(theta_arr)
        assert all(np.abs(diff_theta_arr) < _ANGLE_THRESHOLD), \
            f"Reference path misaligned. Max. allowed orientation difference is {_ANGLE_THRESHOLD}"

    def __getstate__(self):
        """Required for pickling support"""
        return(pycrccosy.CurvilinearCoordinateSystem.__getstate__(self),
               self.__dict__)

    def __setstate__(self, state: tuple):
        """Required for pickling support"""
        pycrccosy.CurvilinearCoordinateSystem.__setstate__(self, state[0])
        self.__dict__ = state[1]

    @property
    def ref_path(self) -> np.ndarray:
        """returns reference path used by CCosy due to slight modifications within the CCosy module"""
        return self._reference_path

    @property
    def ref_pos(self) -> np.ndarray:
        """position (s-coordinate) along reference path"""
        return self._ref_pos

    @property
    def ref_curv(self) -> np.ndarray:
        """curvature along reference path"""
        return self._ref_curv

    @property
    def ref_curv_d(self) -> np.ndarray:
        """curvature rate along reference path"""
        return self._ref_curv_d

    @property
    def ref_theta(self) -> np.ndarray:
        """orientation along reference path"""
        return self._ref_theta

    def plot_reference_states(self):
        """Plots orientation, curvature and curvature rate of ref path over arclength"""
        plt.figure(figsize=(7, 6.5))
        plt.suptitle("Reference path states")
        # orientation
        plt.subplot(3, 1, 1)
        plt.subplot(3, 1, 1)
        plt.plot(self.ref_pos, self.ref_theta, color="k")
        plt.xlabel("s")
        plt.ylabel("orientation")
        # curvature
        plt.subplot(3, 1, 2)
        plt.plot(self.ref_pos, self.ref_curv, color="k")
        plt.xlabel("s")
        plt.ylabel("curvature")
        # curvature rate
        plt.subplot(3, 1, 3)
        plt.plot(self.ref_pos, self.ref_curv_d, color="k")
        plt.xlabel("s")
        plt.ylabel("curvature rate")
        plt.tight_layout()
        plt.show()
