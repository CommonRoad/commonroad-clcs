# third party
import numpy as np

# commonroad-io
from commonroad.common.util import make_valid_orientation

# commonroad-clcs
from commonroad_clcs.util import (
    compute_pathlength_from_polyline,
    compute_orientation_from_polyline,
)


def evaluate_ref_path_curvature_improvements(ref_pos_orig, ref_curv_orig, ref_pos_mod, ref_curv_mod):
    """Evaluates improvements of curvature and curvature rate of original and modified reference path"""
    # compute kappa dots
    ref_curv_d_orig = np.gradient(ref_curv_orig, ref_pos_orig)
    ref_curv_d_mod = np.gradient(ref_curv_mod, ref_pos_mod)

    # compute absolute averages
    # original
    ref_curv_avg_orig = np.average(np.abs(ref_curv_orig))
    ref_curv_avg_mod = np.average(np.abs(ref_curv_mod))

    # modified
    ref_curv_d_avg_orig = np.average(np.abs(ref_curv_d_orig))
    ref_curv_d_avg_mod = np.average(np.abs(ref_curv_d_mod))

    # compute absolute max values
    # original
    ref_curv_max_orig = np.max(np.abs(ref_curv_orig))
    ref_curv_max_mod = np.max(np.abs(ref_curv_mod))

    # modified
    ref_curv_d_max_orig = np.max(np.abs(ref_curv_d_orig))
    ref_curv_d_max_mod = np.max(np.abs(ref_curv_d_mod))

    # compute Deltas
    # average curvature
    delta_curv_avg = np.abs(ref_curv_avg_orig - ref_curv_avg_mod)
    # average curvature rate
    delta_curv_d_avg = np.abs(ref_curv_d_avg_orig - ref_curv_d_avg_mod)

    # maximum curvature
    delta_curv_max = np.abs(ref_curv_max_orig - ref_curv_max_mod)
    delta_curv_d_max = np.abs(ref_curv_d_max_orig - ref_curv_d_max_mod)

    return delta_curv_avg, delta_curv_d_avg, delta_curv_max, delta_curv_d_max


def evaluate_ref_path_deviations(ref_path_orig: np.ndarray, ref_path_mod: np.ndarray, curvilinear_cosy):
    """Evaluates average deviation of pathlength, lateral deviation and orientation deviation"""
    # original
    pathlength_orig = compute_pathlength_from_polyline(ref_path_orig)
    orientation_orig = compute_orientation_from_polyline(ref_path_orig)

    # modified
    pathlength_mod = compute_pathlength_from_polyline(ref_path_mod)
    orientation_mod = compute_orientation_from_polyline(ref_path_mod)

    # list for d and theta deviation
    delta_d_list = list()
    delta_theta_list = list()
    theta_interp_list = list()

    # use ccosy
    for i in range(len(ref_path_orig)):
        vert = ref_path_orig[i]
        vert_converted = curvilinear_cosy.convert_to_curvilinear_coords(vert[0], vert[1])
        s = vert_converted[0]
        d = vert_converted[1]

        delta_d_list.append(d)

        s_idx = np.argmax(pathlength_mod > s) - 1
        if s_idx + 1 >= len(pathlength_mod):
            continue

        theta_interpolated = _interpolate_angle(s,
                                                pathlength_mod[s_idx], pathlength_mod[s_idx + 1],
                                                orientation_mod[s_idx], orientation_mod[s_idx + 1])

        theta_interp_list.append(theta_interpolated)

        delta_theta_list.append(theta_interpolated - orientation_orig[i])

    delta_s = abs(pathlength_orig[-1] - pathlength_mod[-1])
    delta_d_avg = np.average(np.abs(delta_d_list))
    delta_theta_avg = np.average(np.abs(delta_theta_list))

    return delta_s, delta_d_avg, delta_theta_avg


def _interpolate_angle(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Interpolates an angle value between two angles according to the miminal value of the absolute difference
    :param x: value of other dimension to interpolate
    :param x1: lower bound of the other dimension
    :param x2: upper bound of the other dimension
    :param y1: lower bound of angle to interpolate
    :param y2: upper bound of angle to interpolate
    :return: interpolated angular value (in rad)
    """
    delta = y2 - y1
    return make_valid_orientation(delta * (x - x1) / (x2 - x1) + y1)
