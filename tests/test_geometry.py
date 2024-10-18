import unittest
import commonroad_clcs.clcs as clcs
import numpy as np
import pickle
import os


class TestGeometry(unittest.TestCase):

    def setUp(self) -> None:
        try:
            # Local Test side
            with open(os.path.abspath('reference_path_b.pic'), 'rb') as path_file:
                data_set = pickle.load(path_file)
        except OSError as e:
            # CI Test side
            with open(os.path.abspath('geometry/reference_path_b.pic'), 'rb') as path_file:
                data_set = pickle.load(path_file)
        self.reference_path_test = data_set['reference_path']
        self.curvilinear_coord_sys = clcs.CurvilinearCoordinateSystem(self.reference_path_test)

        try:
            # Local Test side
            with open(os.path.abspath('reference_path_property.pic'), 'rb') as property_file:
                property_set = pickle.load(property_file)
        except OSError as e:
            # CI Test side
            with open(os.path.abspath('geometry/reference_path_property.pic'), 'rb') as property_file:
                property_set = pickle.load(property_file)
        self.reference = property_set['reference']
        self.ref_pos = property_set['ref_pos']
        self.ref_curv = property_set['ref_curv']
        self.ref_curv_d = property_set['ref_curv_d']
        self.ref_theta = property_set['ref_theta']

    def test_reference(self):
        assert np.allclose(self.curvilinear_coord_sys.reference, self.reference)

    def test_ref_pos(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_pos, self.ref_pos)

    def test_ref_curv(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_curv, self.ref_curv, atol=1e-04)

    def test_ref_curv_d(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_curv_d, self.ref_curv_d, atol=1e-04)

    def test_ref_theta(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_theta, self.ref_theta)


if __name__ == '__main__':
    unittest.main()
