import unittest
import os

import numpy as np
import pickle

import commonroad_clcs.clcs as clcs
from commonroad_clcs.config import CLCSParams


class TestPyCLCSWrapper(unittest.TestCase):

    def setUp(self) -> None:
        # get path of test directory
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        # get data file
        with open(os.path.join(file_dir_path, 'test_data/reference_path_b.pic'), 'rb') as path_file:
            data_set = pickle.load(path_file)

        # get reference path from test data
        self.reference_path_test = data_set['reference_path']

        # create params objects
        params = CLCSParams()
        params.eps2 = 0.0

        # create CLCS
        self.curvilinear_coord_sys = clcs.CurvilinearCoordinateSystem(
            reference_path=self.reference_path_test,
            params=params,
            preprocess_path=False
        )

        # get data file
        with open(os.path.join(file_dir_path, 'test_data/reference_path_b_data_new.pic'), 'rb') as property_file:
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
