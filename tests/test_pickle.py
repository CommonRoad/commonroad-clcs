import math
import os
import unittest
import pickle

import numpy as np

from commonroad_clcs import pycrccosy
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import CLCSParams


class PickleTest(unittest.TestCase):
    """
    Test cases for testing correct functionality of serialization for pickling the CurvilinearCoordinateSystem
    NOTE: File paths are interpreted differently in the CI side, therefore the following Try-Catch code resolves this
    issue.
    """

    def setUp(self) -> None:
        # get path of test directory
        file_dir_path = os.path.dirname(os.path.realpath(__file__))

        # get data file
        with open(os.path.join(file_dir_path, "test_data/reference_path_b.pic"), "rb") as f:
            data_set = pickle.load(f)

        # get reference path from data
        self.reference_path_test = data_set['reference_path']

        # Instantiate CLCS via pybind C++ object directly
        self.pycrccosy_CLCS = pycrccosy.CurvilinearCoordinateSystem(self.reference_path_test)
        self.pycrccosy_CLCS.compute_and_set_curvature()

        # Instantiate CLCS via Python wrapper class (extends pybind object by additional attributes)
        params = CLCSParams()
        params.default_proj_domain_limit = 20.0
        self.wrapper_CLCS = CurvilinearCoordinateSystem(
            reference_path=self.reference_path_test,
            params=params,
            preprocess_path=False
        )

    def test_pickle_dump_pybind(self):
        clcs_dump = pickle.dumps(self.pycrccosy_CLCS)
        clcs_load = pickle.loads(clcs_dump)
        clcs_load_dump = pickle.dumps(clcs_load)
        assert (clcs_dump == clcs_load_dump)

    def test_pickle_dump_wrapper(self):
        clcs_dump = pickle.dumps(self.wrapper_CLCS)
        clcs_load = pickle.loads(clcs_dump)
        clcs_load_dump = pickle.dumps(clcs_load)
        assert (clcs_dump == clcs_load_dump)

    def test_attributes_after_pickling_pybind(self):
        clcs_dump = pickle.dumps(self.pycrccosy_CLCS)
        clcs_load: pycrccosy.CurvilinearCoordinateSystem = pickle.loads(clcs_dump)

        # type check
        assert type(clcs_load) == pycrccosy.CurvilinearCoordinateSystem

        # reference path
        assert np.allclose(np.asarray(self.pycrccosy_CLCS.reference_path()), np.asarray(clcs_load.reference_path()))
        # reference path original
        assert np.allclose(np.asarray(self.pycrccosy_CLCS.reference_path_original()),
                           np.asarray(clcs_load.reference_path_original()))

        # length
        assert math.isclose(self.pycrccosy_CLCS.length(), clcs_load.length())
        # curvature
        assert np.allclose(np.asarray(self.pycrccosy_CLCS.get_curvature()), np.asarray(clcs_load.get_curvature()))
        # min/max curvature
        assert math.isclose(self.pycrccosy_CLCS.maximum_curvature(), clcs_load.maximum_curvature())
        assert math.isclose(self.pycrccosy_CLCS.maximum_curvature_radius(), clcs_load.maximum_curvature_radius())
        assert math.isclose(self.pycrccosy_CLCS.minimum_curvature(), clcs_load.minimum_curvature())
        assert math.isclose(self.pycrccosy_CLCS.minimum_curvature_radius(), clcs_load.minimum_curvature_radius())

        # projection domains
        assert np.allclose(np.asarray(self.pycrccosy_CLCS.projection_domain()),
                           np.asarray(clcs_load.projection_domain()))

    def test_attributes_after_pickling_wrapper(self):
        clcs_dump = pickle.dumps(self.wrapper_CLCS)
        clcs_load: CurvilinearCoordinateSystem = pickle.loads(clcs_dump)

        # type check
        assert type(clcs_load) == CurvilinearCoordinateSystem

        # *********** Test attributes of pybind base class ****************
        # reference path
        assert np.allclose(np.asarray(self.wrapper_CLCS.reference_path()), np.asarray(clcs_load.reference_path()))
        # reference path original
        assert np.allclose(np.asarray(self.wrapper_CLCS.reference_path_original()),
                           np.asarray(clcs_load.reference_path_original()))

        # length
        assert math.isclose(self.wrapper_CLCS.length(), clcs_load.length())
        # curvature
        assert np.allclose(np.asarray(self.wrapper_CLCS.get_curvature()), np.asarray(clcs_load.get_curvature()))
        # min/max curvature
        assert math.isclose(self.wrapper_CLCS.maximum_curvature(), clcs_load.maximum_curvature())
        assert math.isclose(self.wrapper_CLCS.maximum_curvature_radius(), clcs_load.maximum_curvature_radius())
        assert math.isclose(self.wrapper_CLCS.minimum_curvature(), clcs_load.minimum_curvature())
        assert math.isclose(self.wrapper_CLCS.minimum_curvature_radius(), clcs_load.minimum_curvature_radius())

        # projection domains
        assert np.allclose(np.asarray(self.wrapper_CLCS.projection_domain()),
                           np.asarray(clcs_load.projection_domain()))

        # ************ Test additional attributes of wrapper class *************
        # reference
        assert np.allclose(self.wrapper_CLCS.ref_path, clcs_load.ref_path)
        # reference positions
        assert np.allclose(self.wrapper_CLCS.ref_pos, clcs_load.ref_pos)
        # curvature
        assert np.allclose(self.wrapper_CLCS.ref_curv, clcs_load.ref_curv)
        # curvature rate
        assert np.allclose(self.wrapper_CLCS.ref_curv_d, clcs_load.ref_curv_d)
        # orientation
        assert np.allclose(self.wrapper_CLCS.ref_theta, clcs_load.ref_theta)

    def test_projection_after_pickling_pybind(self):
        # TODO
        clcs_dump = pickle.dumps(self.pycrccosy_CLCS)
        clcs_load: pycrccosy.CurvilinearCoordinateSystem = pickle.loads(clcs_dump)
        pass

    def tets_projection_after_pickling_wrapper(self):
        # TODO
        clcs_dump = pickle.dumps(self.wrapper_CLCS)
        clcs_load: CurvilinearCoordinateSystem = pickle.loads(clcs_dump)
        pass


if __name__ == '__main__':
    unittest.main()
