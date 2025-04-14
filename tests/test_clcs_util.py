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

    def test_resample_polyline(self, intervals=(1.0, 2.0, 3.0), tol=1e-2):
        """Test default user method for polyline resampling with fixed intervals"""
        for interval in intervals:
            # resample
            reference_path_resampled = clcs_util.resample_polyline(self.reference_path_test, interval)

            # check if intervals of resampled polyline are all the same
            seg_intervals = clcs_util.compute_segment_intervals_from_polyline(reference_path_resampled)
            self.assertTrue(
                np.allclose(
                    seg_intervals,
                    seg_intervals[0],
                    atol=tol,
                    rtol=tol
                ),
                msg=f"Intervals after resampling are not the same after resampling with fixed length {interval}"
            )
            # check same start point
            self.assertTrue(
                np.allclose(self.reference_path_test[0], reference_path_resampled[0]),
                msg="Start points of original and resampled polylines should be the same."
            )
            # check same end point
            self.assertTrue(
                np.allclose(self.reference_path_test[-1], reference_path_resampled[-1]),
                msg="End points of original and resampled polylines should be the same."
            )

    def test_resample_polyline_adaptive(self, intervals=((0.4, 1.0), (0.6, 2.0), (0.8, 3.0)), tol=1e-2):
        """Test default user method for polyline resampling with adaptive intervals"""
        for val in intervals:
            # get min and max steps
            min_step = val[0]
            max_step = val[1]

            # resample
            reference_path_resampled = clcs_util.resample_polyline_adaptive(
                self.reference_path_test,
                min_step,
                max_step
            )

            # check intervals of resampled polyline (exclude last point because due to clipping at max length)
            seg_intervals = clcs_util.compute_segment_intervals_from_polyline(reference_path_resampled)
            diff_min = seg_intervals[:-1] - min_step
            diff_max = max_step - seg_intervals[:-1]
            self.assertTrue(
                np.all(diff_min > -tol),
                msg=f"Minimum sampling interval should be {min_step}"
            )
            self.assertTrue(
                np.all(diff_max > -tol),
                msg=f"Maximum sampling interval should be {max_step}"
            )

            # check same start point
            self.assertTrue(
                np.allclose(self.reference_path_test[0], reference_path_resampled[0]),
                msg="Start points of original and resampled polylines should be the same."
            )
            # check same end point
            self.assertTrue(
                np.allclose(self.reference_path_test[-1], reference_path_resampled[-1]),
                msg="End points of original and resampled polylines should be the same."
            )

    def test_resample_polyline_cpp(self):
        """Test own cpp method for resampling"""
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

        # check same start point
        self.assertTrue(np.allclose(self.reference_path_test[0], reference_path_resampled_dense[0]),
                        msg="Start points of original and resampled polylines should be the same.")
        # check same end point
        self.assertTrue(np.allclose(self.reference_path_test[-1], reference_path_resampled_dense[-1]),
                        msg="End points of original and resampled polylines should be the same.")

    def test_resample_polyline_with_length_check(self):
        """Test resampling with length check"""
        length_to_check = 2.0
        reference_path_resampled_length = len(clcs_util.resample_polyline_with_length_check(self.reference_path_test,
                                                                                            length_to_check))

        self.assertGreater(reference_path_resampled_length, self.number_of_samples,
                           msg="The returned polyline should have more samples")

    def test_compute_pathlength_from_polyline(self):
        """Test pathlength computation"""
        returned_path_length = clcs_util.compute_pathlength_from_polyline(self.reference_path_test)
        self.assertEqual(self.number_of_samples, len(returned_path_length),
                         msg='Polylines should be equally resampled')
        assert np.allclose(returned_path_length, self.path_length)

    def test_compute_polyline_length(self):
        """Test length computation"""
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
        """Test orientation computation"""
        returned_orientation = clcs_util.compute_orientation_from_polyline(self.reference_path_test)
        self.assertEqual(self.number_of_samples, len(returned_orientation),
                         msg='Polylines should be equally resampled')
        assert np.allclose(returned_orientation, self.orientation)

    def test_resample_polyline_python(self):
        """Test own python method for polyline resampling"""
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
        """Debug plotting"""
        plt.figure()
        plt.plot(self.reference_path_test[:, 0], self.reference_path_test[:, 1],  marker=".", color="black",
                 label="original")
        plt.plot(ref_path_refined[:, 0], ref_path_refined[:, 1], marker=".", color="red", zorder=20,
                 label="refined")
        plt.show()

    def test_fix_polyline_vertex_ordering(self):
        """Test method for fixing incorrect vertex ordering of a polyline"""
        # threshold
        theta_diff_threshold = np.pi / 2

        # check original path
        theta_arr = clcs_util.compute_orientation_from_polyline(self.reference_path_test)
        diff_theta_arr = np.diff(theta_arr)
        self.assertTrue( all(np.abs(diff_theta_arr) < theta_diff_threshold) )

        # new reference path with wrong ordering
        half_len = int(len(self.reference_path_test) / 2)
        ref_path_faulty = np.vstack(
            [self.reference_path_test[:half_len],
             self.reference_path_test[-1],
             self.reference_path_test[half_len:-1]]
        )

        # check if wrong path is indeed faulty
        theta_arr = clcs_util.compute_orientation_from_polyline(ref_path_faulty)
        diff_theta_arr = np.diff(theta_arr)
        self.assertTrue(not all(np.abs(diff_theta_arr) < theta_diff_threshold))

        # fix path
        ref_path_fixed = clcs_util.fix_polyline_vertex_ordering(
            ref_path_faulty,
            theta_diff_threshold
        )

        # check orientation in fixed path
        theta_arr = clcs_util.compute_orientation_from_polyline(ref_path_fixed)
        diff_theta_arr = np.diff(theta_arr)
        self.assertTrue( all(np.abs(diff_theta_arr) < theta_diff_threshold) )

        # check if fixed path is identical to original path
        self.assertTrue(
            np.allclose(self.reference_path_test, ref_path_fixed),
            msg="Fixed path and original path should be identical."
        )

if __name__ == '__main__':
    unittest.main()
