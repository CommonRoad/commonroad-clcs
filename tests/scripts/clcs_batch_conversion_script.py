import time

import numpy as np
import pickle

from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption
)
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory


# ======== load reference path
with open("../test_data/reference_path_b.pic", "rb") as f:
    data_set = pickle.load(f)
reference_path = data_set['reference_path']


# ======== load points
with open("../test_data/segment_coordinate_system_reference_path_b_points_a.pic", "rb") as f:
    data_set = pickle.load(f)
x = data_set['x']
y = data_set['y']


# ======== smooth curvature
params = CLCSParams()

# pre-process reference path: Here we smooth the path using curve subdivision
params.processing_option = ProcessingOption.CURVE_SUBDIVISION
params.subdivision.degree = 1
params.subdivision.max_curvature = 0.2
params.subdivision.coarse_resampling_step = 1.0
params.resampling.fixed_step = 1.0

ref_path_processor = ProcessorFactory.create_processor(params)
reference_path = ref_path_processor(reference_path)


# ======== create Cosy
cosy = CurvilinearCoordinateSystem(
    reference_path=reference_path,
    params=params,
    preprocess_path=False
)
projection_domain = np.array(cosy.projection_domain())


# ======== Test conversion methods
# NOTE: convert_list_of_points_to_curvilinear_coords will silently drop any points
# outside the projection domain! In order to compare the original and converted
# cartesian points, we need to skip any points outside the projection domain.
# Otherwise the point indices won't line up when comparing the arrays.
points = []
for xv,yv in zip(x,y):
    if cosy.cartesian_point_inside_projection_domain(xv, yv):
        points += [[xv, yv]]
cartesian_points = np.array(points)

# Sanity check that we skipped points outside the projection domain
# Exact number based on test data
assert cartesian_points.shape == (19658, 2), f"Number of points in projection domain: {cartesian_points.shape}"


# Convert points to Curvilinear list conversion
t0 = time.perf_counter()
p_curvilinear = np.array(cosy.convert_list_of_points_to_curvilinear_coords(cartesian_points, 4))
print(f"Cartesian to Curvilinear list conversion took: {time.perf_counter() - t0} s")


# Convert points back to Cartesian list conversion
t0 = time.perf_counter()
p_cartesian = np.array(cosy.convert_list_of_points_to_cartesian_coords(p_curvilinear, 4))
print(f"Curvilinear to Cartesian list conversion took: {time.perf_counter() - t0} s")


# check
assert cartesian_points.shape == p_curvilinear.shape
assert cartesian_points.shape == p_cartesian.shape

# Compare original and converted cartesian coordinates
np.testing.assert_allclose(p_cartesian, cartesian_points, atol=1e-3, rtol=0)


# Convert points to Curvilinear single conversion
list_p_curv = list()
t0 = time.perf_counter()
for pt in cartesian_points:
    p_curv = cosy.convert_to_curvilinear_coords(pt[0], pt[1], check_proj_domain=True)
    list_p_curv.append(p_curv)
print(f"Cartesian to Curvilinear single conversion took: {time.perf_counter() - t0} s")


# Convert points back to Cartesian single conversion
list_p_cart = list()
t0 = time.perf_counter()
for pt in list_p_curv:
    p_cart = cosy.convert_to_cartesian_coords(pt[0], pt[1], check_proj_domain=True)
    list_p_cart.append(p_cart)
print(f"Curvilinear to Cartesian single conversion took: {time.perf_counter() - t0} s")
