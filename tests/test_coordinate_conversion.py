import unittest

import numpy as np

from commonroad_clcs.pycrccosy import (
    CurvilinearCoordinateSystem,
    CartesianProjectionDomainError,
    CurvilinearProjectionDomainLateralError,
    CurvilinearProjectionDomainLongitudinalError
)


class TestCoordinateConversion(unittest.TestCase):
    """
    Base test class for testing coordinate conversions between Cartesian and curvilinear coordinates
    and vice versa
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


class TestSinglePointConversion(TestCoordinateConversion):
    """
    Test class for conversion functions for single points
    """

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


class TestListOfPointsConversion(TestCoordinateConversion):
    """
    Test class for conversion functions for list of points
    """

    pass


class TestPolygonConversion(TestCoordinateConversion):
    """
    Test class for conversion functions for polygons
    """
    pass


class TestRectangleConversion(TestCoordinateConversion):
    """
    Test class for conversion functions for rectangles
    """


if __name__ == '__main__':
    unittest.main()
