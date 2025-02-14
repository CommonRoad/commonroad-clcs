from typing import Optional
import unittest

import numpy as np
import pickle

from commonroad_clcs.pycrccosy import (
    CurvilinearCoordinateSystem,
    CartesianProjectionDomainError,
    CurvilinearProjectionDomainLateralError,
    CurvilinearProjectionDomainLongitudinalError
)
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption
)
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory


class TestSinglePointConversion(unittest.TestCase):
    """
    Test class for conversion functions for single points
    """

    def setUp(self) -> None:
        # plot settings
        self.show_plots = True
        self.plot_points = list()

        # create reference path
        x_coords = np.linspace(0, 5, num=11)
        y_coords = np.zeros(11)
        self.reference_path = np.column_stack((x_coords, y_coords))

        # create CCosy
        self.default_lim = 30.0
        self.eps = 0.1
        self.eps2 = 0.3
        self.ccosy = CurvilinearCoordinateSystem(self.reference_path, self.default_lim, self.eps, self.eps2)

        # get projection domain CART and CURV
        self.proj_domain_cart = np.array(self.ccosy.projection_domain())
        self.proj_domain_curv = np.array(self.ccosy.convert_polygon_to_curvilinear_coords(self.proj_domain_cart))[0]

    def test_convert_to_curvilinear_coordinates(self):
        """
        Conversion from Cartesian to curvilinear
        """
        # test point inside projection domain
        pt_cart_1 = np.array([0.0, 10.0])
        ground_truth = np.array([0.9, 10.0])
        pt_curv = self.ccosy.convert_to_curvilinear_coords(pt_cart_1[0], pt_cart_1[1])
        assert np.allclose(pt_curv, ground_truth)

        # test point on projection domain border
        pt_cart_2 = np.array([-0.6, 30.0])
        ground_truth = np.array([0.3, 30.0])
        pt_curv = self.ccosy.convert_to_curvilinear_coords(pt_cart_2[0], pt_cart_2[1])
        assert np.allclose(pt_curv, ground_truth)

        # test point outside projection domain
        pt_cart_3 = np.array([-1.0, 10.0])
        exception_raised = False
        try:
            pt_curv = self.ccosy.convert_to_curvilinear_coords(pt_cart_3[0], pt_cart_3[1])
        except CartesianProjectionDomainError:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_convert_to_cartesian_coordinates(self):
        """
        Conversion from curvilinear to Cartesian
        """
        # test point inside projection domain
        pt_curv = np.array([5.0, 10.0])
        ground_truth = np.array([4.1, 10.0])
        pt_cart = self.ccosy.convert_to_cartesian_coords(pt_curv[0], pt_curv[1])
        assert np.allclose(pt_cart, ground_truth)

        # test point on projection domain border
        pt_curv = np.array([4.0, 30.0])
        ground_truth = np.array([3.1, 30.0])
        pt_cart = self.ccosy.convert_to_cartesian_coords(pt_curv[0], pt_curv[1])
        assert np.allclose(pt_cart, ground_truth)

        # test longitudinal coordinate outside of reference path
        pt_curv = np.array([-1.0, 10.0])
        exception_raised = False
        try:
            pt_cart = self.ccosy.convert_to_cartesian_coords(pt_curv[0], pt_curv[1])
        except CurvilinearProjectionDomainLongitudinalError:
            exception_raised = True
        self.assertTrue(exception_raised)

        # test lateral coordinate outside of domain
        pt_curv = np.array([6.0, 31.0])
        exception_raised = False
        try:
            pt_cart = self.ccosy.convert_to_cartesian_coords(pt_curv[0], pt_curv[1])
        except CurvilinearProjectionDomainLateralError:
            exception_raised = True
        self.assertTrue(exception_raised)


class TestListOfPointsConversion(unittest.TestCase):
    """
    Test class for conversion functions for list of points
    """
    def setUp(self):
        # load reference path
        with open("./test_data/reference_path_b.pic", "rb") as f:
            data_set = pickle.load(f)
        self.reference_path = data_set['reference_path']

        # load points
        with open("./test_data/segment_coordinate_system_reference_path_b_points_a.pic", "rb") as f:
            data_set = pickle.load(f)
        self.x = data_set['x']
        self.y = data_set['y']

        # process reference path
        params = CLCSParams()
        params.processing_option = ProcessingOption.CURVE_SUBDIVISION
        params.subdivision.degree = 1
        params.subdivision.max_curvature = 0.2
        params.subdivision.coarse_resampling_step = 1.0
        params.resampling.fixed_step = 1.0

        ref_path_processor = ProcessorFactory.create_processor(params)
        self.reference_path = ref_path_processor(self.reference_path)

        # create CLCS
        self.ccosy = CurvilinearCoordinateSystem(
            self.reference_path, 40.0, 0.1, 1e-2
        )

        # cartesian points in projection domain
        self.cart_points_in_proj_domain = self._get_points_in_proj_domain()
        self.num_cart_points_in_proj_domain = self.cart_points_in_proj_domain.shape[0]

    def _get_points_in_proj_domain(self) -> np.ndarray:
        points = []
        for xv,yv in zip(self.x, self.y):
            if self.ccosy.cartesian_point_inside_projection_domain(xv, yv):
                points += [[xv, yv]]
        return np.array(points)

    def test_convert_list_of_points(self):
        """Tests list conversion of points from Cart to Curv and vice-versa"""
        # convert to curvilinear
        curv_points = np.array(self.ccosy.convert_list_of_points_to_curvilinear_coords(
            self.cart_points_in_proj_domain, 4))

        # check same number of points
        self.assertEqual(curv_points.shape[0], self.num_cart_points_in_proj_domain)

        # convert back to Cartesian
        cart_points = np.array(self.ccosy.convert_list_of_points_to_cartesian_coords(curv_points, 4))

        # check same number of points
        self.assertEqual(cart_points.shape[0], self.num_cart_points_in_proj_domain)

        # check same Cartesian points after converting back and forth
        np.testing.assert_allclose(
            cart_points, self.cart_points_in_proj_domain, atol=1e-3, rtol=0
        )

    def test_consistency_list_and_single_conversion(self):
        """Tests consistency between list conversion and single conversion methods"""
        # ======== Cartesian to Curvilinear
        # list conversion
        curv_points_list = np.array(
            self.ccosy.convert_list_of_points_to_curvilinear_coords(self.cart_points_in_proj_domain, 4)
        )
        # single conversion
        curv_points_single = list()
        for pt in self.cart_points_in_proj_domain:
            p_curv = self.ccosy.convert_to_curvilinear_coords(pt[0], pt[1])
            curv_points_single.append(p_curv)
        curv_points_single = np.array(curv_points_single)

        # check if both results are consistent
        np.testing.assert_allclose(
            curv_points_list, curv_points_single, atol=1e-3, rtol=0.0
        )

        # ======== Curvilinear to Cartesian
        cart_points_list = np.array(
            self.ccosy.convert_list_of_points_to_cartesian_coords(curv_points_list, 4)
        )
        # single conversion
        cart_points_single = list()
        for pt in curv_points_single:
            p_cart = self.ccosy.convert_to_cartesian_coords(pt[0], pt[1])
            cart_points_single.append(p_cart)
        cart_points_single = np.array(cart_points_single)

        # check if both results are consistent
        np.testing.assert_allclose(
            cart_points_list, cart_points_single, atol=1e-3, rtol=0.0
        )


if __name__ == '__main__':
    unittest.main()
