import math
import unittest
import os

import numpy as np
import pickle
from matplotlib import pyplot as plt

import commonroad_clcs.util as clcs_util

class TestCLCSUtil(unittest.TestCase):
    def setUp(self) -> None:
        # Debug plot settings (default False, because of CI)
        self.show_plots = False

        # get path of test directory
        file_dir_path = os.path.dirname(os.path.realpath(__file__))

        # get data file
        with open(os.path.join(file_dir_path, 'test_data/reference_path_b.pic'), 'rb') as path_file:
            data_set = pickle.load(path_file)

        # get reference path from data
        self.reference_path_test = data_set['reference_path']
        self.number_of_samples = len(self.reference_path_test)

        # get data file
        with open(os.path.join(file_dir_path, 'test_data/reference_path_b_data_new.pic'), 'rb') as data_file:
            data_details = pickle.load(data_file)

        # get reference path properties from data
        self.polyline_length = data_details['polyline_length']
        self.path_length = data_details['path_length']
        self.curvature = data_details['curvature']
        self.orientation = data_details['orientation']

    def test_resample_polyline_cpp(self):
        # sampling intervals
        interval_dense = 1.0
        interval_original = 2.0
        interval_coarse = 3.0

        reference_path_resampled_dense = clcs_util.resample_polyline_cpp(self.reference_path_test, interval_dense)
        reference_path_resampled_original = clcs_util.resample_polyline_cpp(self.reference_path_test, interval_original)
        reference_path_resampled_coarse = clcs_util.resample_polyline_cpp(self.reference_path_test, interval_coarse)

        # check number of samples
        self.assertGreater(len(reference_path_resampled_dense), self.number_of_samples,
                           msg="Number of samples should be larger")
        self.assertEqual(len(reference_path_resampled_original), self.number_of_samples,
                         msg="Number of samples should be equal")
        self.assertLess(len(reference_path_resampled_coarse), self.number_of_samples,
                        msg="Number of samples should be smaller")

        # check resampled pathlength intervals
        assert np.allclose(clcs_util.compute_segment_intervals_from_polyline(reference_path_resampled_dense)[:-1],
                           interval_dense,
                           atol=1e-02, rtol=1e-02)
        assert np.allclose(clcs_util.compute_segment_intervals_from_polyline(reference_path_resampled_original)[:-1],
                           interval_original,
                           atol=1e-02, rtol=1e-02)
        assert np.allclose(clcs_util.compute_segment_intervals_from_polyline(reference_path_resampled_coarse)[:-1],
                           interval_coarse,
                           atol=1e-02, rtol=1e-02)

    def test_resample_polyline_with_length_check(self):
        length_to_check = 2.0
        reference_path_resampled_length = len(clcs_util.resample_polyline_with_length_check(self.reference_path_test,
                                                                                            length_to_check))

        self.assertGreater(reference_path_resampled_length, self.number_of_samples,
                           msg="The returned polyline should have more samples")

    def test_compute_pathlength_from_polyline(self):
        returned_path_length = clcs_util.compute_pathlength_from_polyline(self.reference_path_test)
        self.assertEqual(self.number_of_samples, len(returned_path_length),
                         msg='Polylines should be equally resampled')
        assert np.allclose(returned_path_length, self.path_length)

    def test_compute_polyline_length(self):
        returned_polyline_length = clcs_util.compute_polyline_length(self.reference_path_test)
        assert math.isclose(returned_polyline_length, self.polyline_length)

    def test_compute_curvature_from_polyline(self):
        """Tests pybind CPP function for curvature computation"""
        curvature_array = clcs_util.compute_curvature_from_polyline(self.reference_path_test)
        self.assertEqual(self.number_of_samples, len(curvature_array),
                         msg='Polylines should be equally resampled')
        assert np.allclose(curvature_array, self.curvature, rtol=1e-3)

    def test_compute_curvature_from_polyline_python(self):
        """Tests consistency between python and cpp curvature computation"""
        curvature_array_cpp = clcs_util.compute_curvature_from_polyline(self.reference_path_test)
        curvature_array_py = clcs_util.compute_curvature_from_polyline_python(self.reference_path_test)
        assert np.allclose(curvature_array_cpp, curvature_array_py, rtol=1e-3)

    def test_compute_orientation_from_polyline(self):
        returned_orientation = clcs_util.compute_orientation_from_polyline(self.reference_path_test)
        self.assertEqual(self.number_of_samples, len(returned_orientation),
                         msg='Polylines should be equally resampled')
        assert np.allclose(returned_orientation, self.orientation)

    def test_resample_polyline_python(self):
        self.assertGreaterEqual(self.number_of_samples, 2,
                                msg="Polyline should have at least 2 points")
        returned_polyline = clcs_util.resample_polyline_python(self.reference_path_test, 2.0)
        test_check = True
        length_to_check = np.linalg.norm(returned_polyline[1] - returned_polyline[0])
        tolerance = 1e-1
        length_to_check_min = length_to_check - tolerance
        length_to_check_max = length_to_check + tolerance
        for i in range(1, len(returned_polyline)):
            length = np.linalg.norm(returned_polyline[i] - returned_polyline[i - 1])
            if length < length_to_check_min or length > length_to_check_max:
                test_check = False
                break
        self.assertEqual(test_check, True,
                         msg="Polyline is not resampled with equidistant spacing")

    def test_chaikins_corner_cutting(self):
        """Curve subdivision using Chaikins corner cutting algorithm"""
        ref_path_refined = clcs_util.chaikins_corner_cutting(self.reference_path_test)
        #  check correct number of points (2*original)
        self.assertEqual(len(ref_path_refined), 2 * len(self.reference_path_test),
                         msg="Refined Polyline should have 2*n points")
        # check same start point
        self.assertTrue(np.allclose(self.reference_path_test[0], ref_path_refined[0]),
                        msg="Start points of original and refined polyline should be the same.")
        # check same end point
        self.assertTrue(np.allclose(self.reference_path_test[-1], ref_path_refined[-1]),
                        msg="End points of original and refined polyline should be the same.")

        if self.show_plots:
            self._plot_subdivision_test(ref_path_refined)

    def test_lane_riesenfeld_subdivision(self):
        """Curve subdivision using Lane Riesenfeld algorithm of degree 2"""
        ref_path_refined = clcs_util.lane_riesenfeld_subdivision(self.reference_path_test,
                                                                 degree=2,
                                                                 refinements=1)
        #  check correct number of points (2*original)
        self.assertEqual(len(ref_path_refined), 2 * len(self.reference_path_test) + 1,
                         msg="Refined Polyline should have 2*n points")
        # check same start point
        self.assertTrue(np.allclose(self.reference_path_test[0], ref_path_refined[0]),
                        msg="Start points of original and refined polyline should be the same.")
        # check same end point
        self.assertTrue(np.allclose(self.reference_path_test[-1], ref_path_refined[-1]),
                        msg="End points of original and refined polyline should be the same.")

        if self.show_plots:
            self._plot_subdivision_test(ref_path_refined)

    def test_consistency_chaikins_and_subdivision(self):
        """Tests consistency of results between Chaikins's algorithm and lr subdivision"""
        # LR algorithm for degree=1 should be consistent with Chaikin's algorithm
        degree = 1
        ref_path_refined_lr = clcs_util.lane_riesenfeld_subdivision(self.reference_path_test,
                                                                    degree=degree,
                                                                    refinements=1)
        ref_path_refined_chaikins = clcs_util.chaikins_corner_cutting(self.reference_path_test,
                                                                      refinements=1)
        # check number of points
        self.assertEqual(len(ref_path_refined_lr), len(ref_path_refined_chaikins),
                         msg="Both refined paths should have the same number of points")
        # check identical points
        self.assertTrue(np.allclose(ref_path_refined_lr, ref_path_refined_chaikins),
                        msg="All points of the refined polylines should be identical")

    def _plot_subdivision_test(self, ref_path_refined):
        plt.figure()
        plt.plot(self.reference_path_test[:, 0], self.reference_path_test[:, 1],  marker=".", color="black",
                 label="original")
        plt.plot(ref_path_refined[:, 0], ref_path_refined[:, 1], marker=".", color="red", zorder=20,
                 label="refined")
        plt.show()


if __name__ == '__main__':
    unittest.main()
