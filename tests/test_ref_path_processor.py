# standard imports
from typing import Tuple
import unittest
import os
import pickle

# third party
import numpy as np

from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs.ref_path_processing.implementation import (
    IReferencePathProcessor,
    NoPreProcessor,
    ResamplingProcessor,
    CurveSubdivisionProcessor,
    SplineSmoothingProcessor,
    ElasticBandProcessor,
)
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption,
    ResamplingOption
)

from commonroad_clcs import util as clcs_util


class RefPathProcessorTest(unittest.TestCase):
    """
    Test cases for the ReferencePathProcessor interface for different options for pre-processing the reference path.
    """

    def setUp(self) -> None:
        """Set up test cases"""
        # get path of test directory
        file_dir_path = os.path.dirname(os.path.realpath(__file__))

        # get data file
        with open(os.path.join(file_dir_path, "test_data/reference_path_b.pic"), "rb") as f:
            data_set = pickle.load(f)

        # get original reference path from data
        self.ref_path_orig = data_set['reference_path']

    def _run_processor(self, params: CLCSParams) -> Tuple[IReferencePathProcessor, np.ndarray]:
        """
        Creates amd runs reference path processor with given params
        Returns tuple of processor object and new reference path
        """
        _ref_path_processor = ProcessorFactory.create_processor(params)
        _ref_path_new = _ref_path_processor(self.ref_path_orig)

        return _ref_path_processor, _ref_path_new

    def test_no_pre_processor(self):
        """Test case for NoPreProcessor"""
        # set params
        params = CLCSParams(processing_option=ProcessingOption.NONE)

        # process reference path
        ref_path_processor, ref_path_new = self._run_processor(params)

        # check
        self.assertIsInstance(ref_path_processor, NoPreProcessor)
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_new)
        )
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_processor.ref_path_original)
        )

    def test_resampling_processor_fixed_sampling(self):
        """Test case for ResamplingProcessor with fixed sampling"""
        # set params
        params = CLCSParams(processing_option=ProcessingOption.RESAMPLE)
        params.resampling.option = ResamplingOption.FIXED
        params.resampling.fixed_step = 1.0

        # process reference path
        ref_path_processor, ref_path_new = self._run_processor(params)

        # check
        self.assertIsInstance(ref_path_processor, ResamplingProcessor)
        # check sampling of output path
        self.assertTrue(
            np.allclose(clcs_util.compute_segment_intervals_from_polyline(ref_path_new),
                        params.resampling.fixed_step,
                        atol=1e-02)
        )
        # check if original path is unchanged
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_processor.ref_path_original)
        )

    def test_resampling_processor_adaptive_sampling(self, atol=1e-2):
        """Test case for ResamplingProcessor with adaptive sampling"""
        # set params
        params = CLCSParams(processing_option=ProcessingOption.RESAMPLE)
        params.resampling.option = ResamplingOption.ADAPTIVE
        params.resampling.min_step = 0.5
        params.resampling.max_step = 2.0

        # process reference path
        ref_path_processor, ref_path_new = self._run_processor(params)

        # check
        self.assertIsInstance(ref_path_processor, ResamplingProcessor)
        # check sampling of output path
        seg_intervals = clcs_util.compute_segment_intervals_from_polyline(ref_path_new)
        diff_min = seg_intervals[:-1] - params.resampling.min_step
        diff_max = params.resampling.max_step - seg_intervals[:-1]
        self.assertTrue(
            np.all(diff_min > -atol),
            msg=f"minimum sampling interval should be {params.resampling.min_step}"
        )
        self.assertTrue(
            np.all(diff_max > -atol),
            msg=f"maximum sampling interval should be {params.resampling.max_step}"
        )
        # check if original path is unchanged
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_processor.ref_path_original)
        )

    def test_curve_subdivision_processor(self):
        """Test case for CurveSubdivisionProcessor"""
        # get current  max curvature
        curr_max_curv = np.max(clcs_util.compute_curvature_from_polyline_python(self.ref_path_orig))

        # set params
        params = CLCSParams(processing_option=ProcessingOption.CURVE_SUBDIVISION)
        params.subdivision.degree = 2
        params.resampling.option = ResamplingOption.FIXED
        params.resampling.fixed_step = 1.0
        params.subdivision.max_curvature = 0.75 * curr_max_curv

        # process reference path
        ref_path_processor, ref_path_new = self._run_processor(params)

        # check
        self.assertIsInstance(ref_path_processor, CurveSubdivisionProcessor)
        # check max curvature
        self.assertLessEqual(
            np.max(clcs_util.compute_curvature_from_polyline_python(ref_path_new)),
            params.subdivision.max_curvature
        )
        # check sampling of output path
        self.assertTrue(
            np.allclose(clcs_util.compute_segment_intervals_from_polyline(ref_path_new),
                        params.resampling.fixed_step,
                        atol=1e-02)
        )
        # check if original path is unchanged
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_processor.ref_path_original)
        )

    def test_spline_smoothing_processor(self):
        """Test case for SplineSmoothingProcessor"""
        # set params
        params = CLCSParams(processing_option=ProcessingOption.SPLINE_SMOOTHING)
        params.spline.smoothing_factor = 10.0
        params.resampling.option = ResamplingOption.FIXED
        params.resampling.fixed_step = 2.0

        # process reference path
        ref_path_processor, ref_path_new = self._run_processor(params)

        # check
        self.assertIsInstance(ref_path_processor, SplineSmoothingProcessor)
        # check max curvature
        self.assertLess(
            np.max(clcs_util.compute_curvature_from_polyline_python(ref_path_new)),
            np.max(clcs_util.compute_curvature_from_polyline_python(self.ref_path_orig))
        )
        # check sampling of output path
        self.assertTrue(
            np.allclose(clcs_util.compute_segment_intervals_from_polyline(ref_path_new),
                        params.resampling.fixed_step,
                        atol=1e-02)
        )
        # check if original path is unchanged
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_processor.ref_path_original)
        )

    def test_elastic_band_processor(self):
        """Test case for ElasticBandProcessor"""
        # set params
        params = CLCSParams(processing_option=ProcessingOption.ELASTIC_BAND)
        params.elastic_band.input_resampling_step = 2.0
        params.resampling.option = ResamplingOption.FIXED
        params.resampling.fixed_step = 2.0

        # process reference path
        ref_path_processor, ref_path_new = self._run_processor(params)

        # check
        self.assertIsInstance(ref_path_processor, ElasticBandProcessor)
        # check max curvature
        self.assertLess(
            np.max(clcs_util.compute_curvature_from_polyline_python(ref_path_new)),
            np.max(clcs_util.compute_curvature_from_polyline_python(self.ref_path_orig)),
        )
        # check sampling of output path
        self.assertTrue(
            np.allclose(clcs_util.compute_segment_intervals_from_polyline(ref_path_new),
                        params.resampling.fixed_step,
                        atol=1e-02)
        )
        # check if original path is unchanged
        self.assertTrue(
            np.array_equal(self.ref_path_orig, ref_path_processor.ref_path_original)
        )
