# standard imports
from typing import Optional

# third party
import numpy as np

from .processor import IReferencePathProcessor

from commonroad_clcs.util import (
    resample_polyline,
    resample_polyline_adaptive,
)

from commonroad_clcs.helper.smoothing import (
    smooth_polyline_subdivision,
    smooth_polyline_spline,
    smooth_polyline_elastic_band
)

from commonroad_clcs.config import (
    ResamplingParams,
    ResamplingOption
)


class NoPreProcessor(IReferencePathProcessor):
    """No pre-processing of the reference path"""

    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        return ref_path_input


class ResamplingProcessor(IReferencePathProcessor):
    """Pre-processing strategy for only resampling the reference path with the given resampling options"""

    @classmethod
    def resample_path(cls, ref_path: np.ndarray, resampling_params: ResamplingParams) -> Optional[np.ndarray]:
        """Resample path with the given options."""
        # fixed resampling
        if resampling_params.option == ResamplingOption.FIXED:
            step = resampling_params.fixed_step
            return resample_polyline(ref_path, step)
        # curvature adaptive resampling
        elif resampling_params.option == ResamplingOption.ADAPTIVE:
            min_step = resampling_params.min_step
            max_step = resampling_params.max_step
            return resample_polyline_adaptive(ref_path,
                                              min_ds=min_step,
                                              max_ds=max_step,
                                              interpolation_type=resampling_params.interpolation_type
                                              )

    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        return self.resample_path(ref_path_input, self._params.resampling)


class CurveSubdivisionProcessor(IReferencePathProcessor):
    """Pre-processing strategy for smoothing via curve-subdivision"""

    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        # get params
        subdiv_params = self._params.subdivision
        resampling_params = self._params.resampling
        _verbose = kwargs.get("verbose") if kwargs.get("verbose") is not None else False

        # curve subdivision smoothing
        new_polyline = smooth_polyline_subdivision(ref_path_input,
                                                   degree=subdiv_params.degree,
                                                   refinements=subdiv_params.num_refinements,
                                                   coarse_resampling=subdiv_params.coarse_resampling_step,
                                                   max_curv=subdiv_params.max_curvature,
                                                   max_dev=subdiv_params.max_deviation,
                                                   max_iter=subdiv_params.max_iterations,
                                                   verbose=_verbose)

        # postprocess: resample final polyline according to resampling options
        return ResamplingProcessor.resample_path(new_polyline, resampling_params)


class SplineSmoothingProcessor(IReferencePathProcessor):
    """Pre-processing strategy for smoothing by fitting an approximating B-spline to the original polyline"""

    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        # get params
        spline_params = self._params.spline
        resampling_params = self._params.resampling

        # spline smoothing
        new_polyline = smooth_polyline_spline(ref_path_input,
                                              step=None,
                                              degree=spline_params.degree_spline,
                                              smoothing_factor=spline_params.smoothing_factor)

        # resample path for final ref path
        return ResamplingProcessor.resample_path(new_polyline, resampling_params)


class ElasticBandProcessor(IReferencePathProcessor):
    """Pre-processing strategy for smoothing via Elastic Band optimization"""

    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        # get params
        eb_params = self._params.elastic_band
        resampling_params = self._params.resampling

        # elastic band smoothing
        new_polyline = smooth_polyline_elastic_band(polyline=ref_path_input,
                                                    max_deviation=eb_params.max_deviation,
                                                    weight_smooth=eb_params.weight_smooth,
                                                    weight_lat_error=eb_params.weight_lat_error,
                                                    solver_max_iter=eb_params.max_iteration)

        # resample path for final ref path
        return ResamplingProcessor.resample_path(new_polyline, resampling_params)


class MapCoverageProcessor(IReferencePathProcessor):
    """Pre-processing strategy for modifying reference path to maximize map coverage"""

    def _process_path(self, ref_path_input: np.ndarray, *args, **kwargs) -> Optional[np.ndarray]:
        pass
