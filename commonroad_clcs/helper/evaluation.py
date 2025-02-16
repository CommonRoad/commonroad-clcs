# standard imports
from typing import Optional, Dict

# third party
import numpy as np
from matplotlib import pyplot as plt

# commonroad-io
from commonroad.common.util import make_valid_orientation

# commonroad-clcs
from commonroad_clcs import pycrccosy
from commonroad_clcs.util import (
    compute_pathlength_from_polyline,
    compute_orientation_from_polyline,
    compute_curvature_from_polyline_python,
)


def plot_ref_path_curvature(
        reference_path: np.ndarray,
        axs=None,
        label: Optional[str] = None,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
        savepath: Optional[str] = None
) -> None:
    """
    Plots curvature and curvature derivative for a given reference path
    :param reference_path: 2d numpy array
    :param axs: Matplotlib axis object (if provided as argument)
    :param label: plot label
    :param color: plot color
    :param linestyle: plot linestyle
    :param savepath: full path to save figure
    """
    if axs is None:
        fig, axs = plt.subplots(2)

    # get reference states
    # pathlength
    ref_pos = compute_pathlength_from_polyline(reference_path)
    # curvature
    ref_curv = compute_curvature_from_polyline_python(reference_path)
    # curvature derivative
    ref_curv_d = np.gradient(ref_curv, ref_pos)

    # plot curvature
    axs[0].plot(ref_pos, ref_curv, label=label, color=color, linestyle=linestyle)
    axs[0].set(xlabel="$s$", ylabel="$\kappa$")

    # plot curvature 1st derivative
    axs[1].plot(ref_pos, ref_curv_d, label=label, color=color, linestyle=linestyle)
    axs[1].set(xlabel="$s$", ylabel="$\dot{\kappa}$")

    # set legend
    axs[0].legend()

    if savepath:
        plt.axis('on')
        plt.savefig(savepath, format="svg", bbox_inches="tight", transparent=False)


def compare_ref_path_curvatures(
        ref_path_original: np.ndarray,
        ref_path_modified: np.ndarray,
        verbose: bool = False
) -> Dict:
    """
    Compares curvature and curvature rates of an original and modified reference path
    :param ref_path_original: Original reference path
    :param ref_path_modified: Modified reference path
    :param verbose: prints metrics to console
    :return Dictionary with metrics
    """
    # pathlength
    ref_pos_orig = compute_pathlength_from_polyline(ref_path_original)
    ref_pos_mod = compute_pathlength_from_polyline(ref_path_modified)

    # curvature
    ref_curv_orig = compute_curvature_from_polyline_python(ref_path_original)
    ref_curv_mod = compute_curvature_from_polyline_python(ref_path_modified)

    # curvature derivative
    ref_curv_d_orig = np.gradient(ref_curv_orig, ref_pos_orig)
    ref_curv_d_mod = np.gradient(ref_curv_mod, ref_pos_mod)

    # absolute curvature average
    ref_curv_avg_orig = np.average(np.abs(ref_curv_orig))
    ref_curv_avg_mod = np.average(np.abs(ref_curv_mod))

    # absolute curvature derivative average
    ref_curv_d_avg_orig = np.average(np.abs(ref_curv_d_orig))
    ref_curv_d_avg_mod = np.average(np.abs(ref_curv_d_mod))

    # absolut max curvature
    ref_curv_max_orig = np.max(np.abs(ref_curv_orig))
    ref_curv_max_mod = np.max(np.abs(ref_curv_mod))

    # absolute max curvauture derivative
    ref_curv_d_max_orig = np.max(np.abs(ref_curv_d_orig))
    ref_curv_d_max_mod = np.max(np.abs(ref_curv_d_mod))

    # delta average curvature
    delta_curv_avg = np.abs(ref_curv_avg_orig - ref_curv_avg_mod)
    # delta average curvature rate
    delta_curv_d_avg = np.abs(ref_curv_d_avg_orig - ref_curv_d_avg_mod)
    # delta maximum curvature
    delta_curv_max = np.abs(ref_curv_max_orig - ref_curv_max_mod)
    # delta maximum curvature derivative
    delta_curv_d_max = np.abs(ref_curv_d_max_orig - ref_curv_d_max_mod)

    # result dictionary
    metrics_dict = {
        "delta_kappa_avg": delta_curv_avg,
        "delta_kappa_dot_avg": delta_curv_d_avg,
        "delta_kappa_max": delta_curv_max,
        "delta_kappa_dot_max": delta_curv_d_max
    }

    # print to console
    if verbose:
        for k, v in metrics_dict.items():
            print(f"\t {k}: \t {v}")

    return metrics_dict


def compare_ref_path_deviations(
        ref_path_original: np.ndarray,
        ref_path_modified: np.ndarray,
        verbose: bool = False
) -> Dict:
    """
    Computes deviation metrics of a modified reference path to its original reference path.
    --------
    Metrics:
        - delta_s: change in overall path length
        - delta_d_avg: average lateral deviation
        - delta_d_max: maximum (absolute) lateral deviation
        - delta_theta_avg: average orientation deviation
        - delta_theta_max: maximum (absolute) orientation deviation

    :param ref_path_original: Original reference path
    :param ref_path_modified: Modified reference path
    :param verbose: prints metrics to console
    :return Dictionary with metrics
    """
    # original pathlength and orientation
    pathlength_orig = compute_pathlength_from_polyline(ref_path_original)
    orientation_orig = compute_orientation_from_polyline(ref_path_original)

    # modified pathlength and orientation
    pathlength_mod = compute_pathlength_from_polyline(ref_path_modified)
    orientation_mod = compute_orientation_from_polyline(ref_path_modified)

    # list for d and theta deviation
    delta_d_list = list()
    delta_theta_list = list()

    # construct curvilinear coordinate system for modified reference path
    _settings = {
        "default_limit": 40.0,
        "eps": 0.1,
        "eps2": 2.0,
        "logging_level": "off",
        "method": 2
    }
    curvilinear_cosy = pycrccosy.CurvilinearCoordinateSystem(
        ref_path_modified,
        default_projection_domain_limit=40.0,
        eps=0.1,
        eps2=2.0,
        log_level="off",
        method=2
    )

    # calculate lateral deviation and orientation deviation
    for i in range(len(ref_path_original)):
        vert = ref_path_original[i]
        vert_converted = curvilinear_cosy.convert_to_curvilinear_coords(vert[0], vert[1])
        s = vert_converted[0]
        d = vert_converted[1]

        delta_d_list.append(d)

        s_idx = np.argmax(pathlength_mod > s) - 1
        if s_idx + 1 >= len(pathlength_mod):
            continue

        theta_interpolated = _interpolate_angle(
            s,
            pathlength_mod[s_idx],
            pathlength_mod[s_idx + 1],
            orientation_mod[s_idx],
            orientation_mod[s_idx + 1]
        )

        delta_theta_list.append(theta_interpolated - orientation_orig[i])

    delta_s = abs(pathlength_orig[-1] - pathlength_mod[-1])
    delta_d_avg = np.average(np.abs(delta_d_list))
    delta_d_max = np.max(np.abs(delta_d_list))
    delta_theta_avg = np.average(np.abs(delta_theta_list))
    delta_theta_max = np.max(np.abs(delta_theta_list))

    # metrics dictionary
    metrics_dict = {
        "delta_s": delta_s,
        "delta_d_avg": delta_d_avg,
        "delta_d_max": delta_d_max,
        "delta_theta_avg": delta_theta_avg,
        "delta_theta_max": delta_theta_max
    }

    # print to console
    if verbose:
        for k, v in metrics_dict.items():
            print(f"\t {k}: \t {v}")

    return metrics_dict


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
