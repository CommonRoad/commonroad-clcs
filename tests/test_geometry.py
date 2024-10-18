import unittest
import commonroad_clcs.clcs as clcs
import numpy as np
import pickle
import os


class TestGeometry(unittest.TestCase):

    def setUp(self) -> None:
        # get path of test directory
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Local
            with open(os.path.join(file_dir_path, 'test_data/reference_path_b.pic'), 'rb') as path_file:
                data_set = pickle.load(path_file)
        except OSError as e:
            # CI
            with open(os.path.abspath('geometry/reference_path_b.pic'), 'rb') as path_file:
                data_set = pickle.load(path_file)

        # get reference path from test data
        self.reference_path_test = data_set['reference_path']
        self.curvilinear_coord_sys = clcs.CurvilinearCoordinateSystem(self.reference_path_test,
                                                                      eps2=0.0,
                                                                      resample=False)

        try:
            # Local
            with open(os.path.join(file_dir_path, 'test_data/reference_path_b_data_new.pic'), 'rb') as property_file:
                property_set = pickle.load(property_file)
        except OSError as e:
            # CI
            with open(os.path.abspath('geometry/reference_path_property.pic'), 'rb') as property_file:
                property_set = pickle.load(property_file)

        # get reference path properties from data
        self.ref_pos = property_set['path_length']
        self.ref_curv = property_set['curvature']
        self.ref_theta = property_set['orientation']

    def test_ref_pos(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_pos, self.ref_pos)

    def test_ref_curv(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_curv, self.ref_curv, atol=1e-04)

    def test_ref_theta(self):
        assert np.allclose(self.curvilinear_coord_sys.ref_theta, self.ref_theta)


if __name__ == '__main__':
    unittest.main()
