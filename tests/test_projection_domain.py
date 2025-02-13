import unittest

import numpy as np
from matplotlib import pyplot as plt
from parameterized import parameterized_class

from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem
from commonroad_clcs.util import compute_pathlength_from_polyline


@parameterized_class(("default_lim", "method"), [
    (30.0, 1),
    (30.0, 2)
])
class TestProjectionDomain(unittest.TestCase):
    """
    Test class for testing computation and correct functionality of the projection domain of the CLCS
    """

    def setUp(self) -> None:
        # Debug plot settings (default False, because of CI)
        self.show_plots = False

        # create reference path
        self.num_points = 11
        x_coords = np.linspace(0, 5, num=self.num_points)
        y_coords = np.zeros(self.num_points)
        self.reference_path = np.column_stack((x_coords, y_coords))

        # create CCosy
        self.eps = 0.1
        self.eps2 = 0.3
        self.log_level = "off"
        self.ccosy = CurvilinearCoordinateSystem(
            self.reference_path,
            self.default_lim,
            self.eps,
            self.eps2,
            self.log_level,
            self.method
        )

        # get projection domain border CART and CURV
        self.cart_proj_domain_border = np.asarray(self.ccosy.projection_domain())
        self.curv_proj_domain_border = np.asarray(self.ccosy.curvilinear_projection_domain())

        # number of segments
        self.ref_path_ccosy = np.asarray(self.ccosy.reference_path())
        self.num_segments = len(self.ref_path_ccosy) - 1

        # pathlength of CLCS
        self.pathlength = compute_pathlength_from_polyline(self.ref_path_ccosy)

    def test_projection_domain_border(self):
        """Construction of Cartesian projection domain"""
        # test number of points
        num_vertices = len(self.cart_proj_domain_border)
        num_vertices_target = 2 * (self.num_segments - 1) + 1
        self.assertEqual(num_vertices, num_vertices_target)

        # test if polyline is closed (first and last vertex coincide)
        self.assertTrue(np.array_equal(self.cart_proj_domain_border[0], self.cart_proj_domain_border[-1]))

        # plot
        if self.show_plots:
            plt.figure()
            plt.title("Cartesian Projection Domain")
            plt.plot(self.ref_path_ccosy[:, 0], self.ref_path_ccosy[:, 1], color='g', marker='x', markersize=5,
                     zorder=19, linewidth=2.0)
            plt.plot(self.cart_proj_domain_border[:, 0], self.cart_proj_domain_border[:, 1],
                     zorder=100, marker=".", color='orange')
            plt.show()

    def test_upper_projection_domain_border(self):
        """Correctness of the sign of upper projection domain border"""
        # upper projection domain represents "left" side of the path, i.e., positive sign
        upper_border = np.asarray(self.ccosy.upper_projection_domain_border())
        # check if y coordinates have positive sign
        for vert in upper_border:
            y_coord = vert[1]
            self.assertGreater(y_coord, 0.0)

    def test_lower_projection_domain_border(self):
        """Correctness of the sign of the lower projection domain border"""
        # lower projection domain represents the "right" side of the path, i.e., negative sign
        lower_border = np.asarray(self.ccosy.lower_projection_domain_border())
        # check if y coordinates have negative sign
        for vert in lower_border:
            y_coord = vert[1]
            self.assertLess(y_coord, 0.0)

    def test_curvilinear_projection_domain_border(self):
        """Construction of curvilinear projection domain"""
        # test number of points
        num_vertices = len(self.curv_proj_domain_border)
        num_vertices_target = 2 * (self.num_segments - 1) + 1
        self.assertEqual(num_vertices, num_vertices_target)

        # test if polyline is closed (first and last vertex coincide)
        self.assertTrue(np.array_equal(self.curv_proj_domain_border[0], self.curv_proj_domain_border[-1]))

        # test d coordinate (ref path is straight line -> all d values should be the default limit)
        self.assertTrue(np.all((self.curv_proj_domain_border[:, 1] == self.default_lim) |
                               (self.curv_proj_domain_border[:, 1] == -self.default_lim)))

        if self.show_plots:
            plt.figure()
            plt.title("Curvilinear Projection Domain")
            plt.plot(self.pathlength, np.zeros(len(self.pathlength)),
                     zorder=100, marker=".", color='green')
            plt.plot(self.curv_proj_domain_border[:, 0], self.curv_proj_domain_border[:, 1],
                     zorder=100, marker=".", color='orange')
            plt.show()

    def test_convert_cart_to_curv_proj_domain(self):
        """Test correctness of projection domain computation by converting Cartesian border to Curvilinear"""
        for vert in self.cart_proj_domain_border:
            vert_curv = np.asarray(self.ccosy.convert_to_curvilinear_coords(vert[0], vert[1]))
            _tmp = self.curv_proj_domain_border - vert_curv
            _norms = np.linalg.norm(_tmp, axis=1)
            self.assertTrue(np.any(np.isclose(_norms, 0.0)),
                            msg=f"Cartesian vertex {vert} not contained in Curvilinear proj domain after "
                                f"conversion to {vert_curv}")

    def test_convert_curv_to_cart_proj_domain(self):
        """Test correctness of projection domain computation by converting Curvilinear border to Cartesian"""
        for vert in self.curv_proj_domain_border:
            vert_cart = np.asarray(self.ccosy.convert_to_cartesian_coords(vert[0], vert[1]))
            _tmp = self.cart_proj_domain_border - vert_cart
            _norms = np.linalg.norm(_tmp, axis=1)
            self.assertTrue(np.any(np.isclose(_norms, 0.0)),
                            msg=f"Curvilinear vertex {vert} not contained in Cartesian proj domain after "
                                f"conversion to {vert_cart}")

    def test_cart_point_in_projection_domain(self):
        """Tests if Cartesian points are within/outside/on the projection domain border"""
        # point within border
        pt_cart_in = np.array([1.0, 10.0])
        # point outside border
        pt_cart_out = np.array([1.0, 35.0])
        # point on border
        pt_cart_on = np.array([1.0, 30.0])

        # check containments
        _res = self.ccosy.cartesian_point_inside_projection_domain(pt_cart_in[0], pt_cart_in[1])
        self.assertTrue(_res)

        _res = self.ccosy.cartesian_point_inside_projection_domain(pt_cart_out[0], pt_cart_out[1])
        self.assertFalse(_res)

        _res = self.ccosy.cartesian_point_inside_projection_domain(pt_cart_on[0], pt_cart_on[1])
        self.assertTrue(_res)

    def test_curv_point_in_projection_domain(self):
        """Tests if Curvilinear points are within/outside/on the projection domain border"""
        # point within border
        pt_curv_in = np.array([1.0, 10.0])
        # point outside border
        pt_curv_out = np.array([1.0, 35.0])
        # point on border
        pt_curv_on = np.array([1.0, 30.0])

        # check containments
        _res = self.ccosy.curvilinear_point_inside_projection_domain(pt_curv_in[0], pt_curv_in[1])
        self.assertTrue(_res)

        _res = self.ccosy.curvilinear_point_inside_projection_domain(pt_curv_out[0], pt_curv_out[1])
        self.assertFalse(_res)

        _res = self.ccosy.curvilinear_point_inside_projection_domain(pt_curv_on[0], pt_curv_on[1])
        self.assertTrue(_res)

    def test_cart_polygon_subset_within_proj_domain(self):
        """Test subset of a Cartesian polygon which is within the projection domain"""
        pass

    def test_curv_polygon_subset_within_proj_domain(self):
        """Test subset of a Curvilinear polygon which is within the projection domain"""
        pass

