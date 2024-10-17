# standard imports
import math
from typing import List, Optional, Tuple, Callable, Dict
from copy import deepcopy

# third party
import Polygon.Utils as gpc_utils
import numpy as np
from scipy.integrate import quad
from scipy import sparse
from scipy.interpolate import (
    splprep,
    splev,
    interp1d,
    CubicSpline,
    Akima1DInterpolator
)
import osqp

# commonroad
from commonroad.common.validity import is_valid_polyline

# commonroad-dc
import commonroad_clcs.pycrccosy as pycrccosy


def intersect_segment_segment(
        segment_1: Tuple[np.ndarray, np.ndarray],
        segment_2: Tuple[np.ndarray, np.ndarray]
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Checks if two segments intersect; if yes, returns their intersection point

    :param segment_1: Tuple with start and end point of first segment
    :param segment_2: Tuple with start and end point of second segment
    """
    return pycrccosy.Util.intersection_segment_segment(segment_1[0], segment_1[1], segment_2[0], segment_2[1])


def chaikins_corner_cutting(
        polyline: np.ndarray,
        refinements: int = 1
) -> np.ndarray:
    """
    Chaikin's corner cutting algorithm to smooth a polyline by replacing each original point with two new points.
    The new points are at 1/4 and 3/4 along the way of an edge.
    The limit curve of Chaikin's algorithm is a quadratic B-spline with C^1 continuity.

    :param polyline: polyline with 2D points
    :param refinements: how many times apply the chaikins corner cutting algorithm
    :return: smoothed polyline
    """
    assert is_valid_polyline(polyline) and len(polyline) >= 3, "Provided polyline is invalid!"
    new_polyline = pycrccosy.Util.chaikins_corner_cutting(polyline, refinements)
    return np.array(new_polyline)


def lane_riesenfeld_subdivision(
        polyline: np.ndarray,
        degree: int = 2,
        refinements: int = 1
) -> np.ndarray:
    """
    General Lane Riesenfeld curve subdivision algorithm.
    The limit curve of the subdivision with the given degree is a B-spline of degree "degree+1".
    Examples:
    - For degree=2 the limit curve is a cubic B-spline with C^2 continuity.
    - For degree=1, the algorithm corresponds to Chaikin's algorithm and the limit curve is a qudratic B-spine (C^1).
    Note: The resulting polyline has more points than the original polyline.

    :param polyline: polyline with 2D points
    :param degree: degree of subdivision
    :param refinements: number of subdivision refinements
    :return: refined polyline
    """
    assert is_valid_polyline(polyline) and len(polyline) >= 3, "Provided polyline is invalid!"
    new_polyline = pycrccosy.Util.lane_riesenfeld_subdivision(polyline, degree, refinements)
    return np.array(new_polyline)


def compute_pathlength_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the path length of a given polyline

    :param polyline: polyline with 2D points
    :return: path length of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for pathlength computation p={}'.format(polyline)
    distance = [0]
    for i in range(1, len(polyline)):
        distance.append(distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1]))
    return np.array(distance)


def compute_polyline_length(polyline: np.ndarray) -> float:
    """
    Computes the length of a given polyline

    :param polyline: The polyline
    :return: The path length of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(polyline[:,0]) > 2, \
        'Polyline malformed for path length computation p={}'.format(polyline)

    distance_between_points = np.diff(polyline, axis=0)
    # noinspection PyTypeChecker
    return np.sum(np.sqrt(np.sum(distance_between_points ** 2, axis=1)))


def compute_segment_intervals_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Compute the interval length of each segment of the given polyline.

    :param polyline: input polyline
    :return: array with interval lengths for polyline segments.
    """
    # compute pathlength
    pathlength = compute_pathlength_from_polyline(polyline)
    return np.diff(pathlength)


def compute_curvature_from_polyline(polyline: np.ndarray, digits: int = 8) -> np.ndarray:
    """
    Computes the curvature of a given polyline. It is assumed that he points of the polyline are sampled equidistant.

    :param polyline: The polyline for the curvature computation
    :param digits: precision for curvature computation
    :return: The curvature of the polyline
    """
    assert is_valid_polyline(polyline) and len(polyline) >= 3, "Polyline p={} is malformed!".format(polyline)

    curvature = pycrccosy.Util.compute_curvature(polyline, digits)
    return curvature


def compute_curvature_from_polyline_python(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvatures along a given polyline

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Curvature array of the polyline for each coordinate [1/rad]
    """
    assert is_valid_polyline(polyline) and len(polyline) >= 3, "Polyline p={} is malformed!".format(polyline)
    pathlength = compute_pathlength_from_polyline(polyline)

    # compute first and second derivatives
    x_d = np.gradient(polyline[:, 0], pathlength)
    x_dd = np.gradient(x_d, pathlength)
    y_d = np.gradient(polyline[:, 1], pathlength)
    y_dd = np.gradient(y_d, pathlength)

    return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))


def compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the orientation of a given polyline

    :param polyline: polyline with 2D points
    :return: orientation of polyline
    """
    assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(polyline[0, :]) == 2, \
        'not a valid polyline. polyline = {}'.format(polyline)

    if len(polyline) < 2:
        raise NameError('Cannot create orientation from polyline of length < 2')

    orientation = []
    for i in range(0, len(polyline) - 1):
        pt1 = polyline[i]
        pt2 = polyline[i + 1]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    for i in range(len(polyline) - 1, len(polyline)):
        pt1 = polyline[i - 1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    return np.array(orientation)


def get_inflection_points(polyline: np.ndarray, digits: int = 4) -> Tuple[np.ndarray, List]:
    """
    Returns the inflection points (i.e., points where the sign of curvature changes) of a polyline
    :param polyline: discrete polyline with 2D points
    :param digits: precision for curvature computation to identify inflection points
    :return: tuple (inflection points, list of indices)
    """
    idx_inflection_points = pycrccosy.Util.get_inflection_point_idx(polyline, digits)
    return polyline[idx_inflection_points], idx_inflection_points


def chord_error_arc(curvature: float, seg_length: float) -> float:
    """
    Computes the chord error when approximating a circular arc of given curvature with linear segments.
    :param curvature: curvature of the circular arc
    :param seg_length: length of the linear segment
    :return: chord error of approximation
    """
    return 1 / curvature * (1 - math.cos(0.5 * curvature * seg_length))


def smooth_polyline_subdivision(
        polyline: np.ndarray,
        degree: int,
        refinements: int = 3,
        coarse_resampling: float = 2.0,
        max_curv: Optional[float] = None,
        max_dev: Optional[float] = None,
        max_iter: Optional[int] = None
) -> np.ndarray:
    """
    Smooths a polyline using Lane-Riesenfeld curve subdivision.
    The degree of the subdivision k has the B-spline curve of degree k+1 as it's limit curve.
    E.g., for degree k=2 the subdivision converges to a cubic B-spline with C^2 continuity.
    This function iteratively reduces the curvature and smooths the input polyline.

    :param polyline: input polyline
    :param degree: curve subdivision degree
    :param refinements: number of subdivision steps
    :param coarse_resampling:
    :param max_curv: maximum curvature for smoothing
    :param max_dev: maximum deviation from ref path
    :param max_iter: maximum number of smoothing iterations
    :return: smoothed polyline as np.ndarray
    """
    new_polyline = deepcopy(polyline)

    # get max curvature
    max_curv = max_curv if max_curv is not None else 10.0

    # iteration counter
    iter_cnt = 0

    # current maximum curvature
    curr_max_curv = np.max(compute_curvature_from_polyline_python(new_polyline))

    # iterative smoothing
    # TODO use max deviation ?
    while (curr_max_curv > max_curv) or (iter_cnt < max_iter):
        new_polyline = lane_riesenfeld_subdivision(new_polyline, degree, refinements)

        # get max curvature
        curr_max_curv = np.max(compute_curvature_from_polyline_python(new_polyline))

        # resample with coarse step
        new_polyline = resample_polyline(new_polyline, coarse_resampling)
        iter_cnt += 1

    # postprocess: refine final polyline for smoothness
    new_polyline = lane_riesenfeld_subdivision(new_polyline, degree, refinements)

    return new_polyline


def smooth_polyline_spline(
        polyline: np.ndarray,
        degree: int = 3,
        smoothing_factor: float = 0.0,
        step: Optional[float] = None
) -> np.ndarray:
    """
    Smooths a polyline via spline interpolation.
    See scipy.splprep for details: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
    :param polyline: discrete polyline with 2D points
    :param step: sampling interval for resampling of final polyline
    :param degree: degree of the B-Spline (default cubic spline)
    :param smoothing_factor: tradeoff between closeness of fit and smoothness
    :return final polyline determined by fitting a smoothing B-spline
    """
    # scipy.splprep (procedural, parametric)
    u = np.linspace(0, 1, len(polyline[:, 0]))  # uniform parametrization: equivalent to setting u=None
    tck, u = splprep(polyline.T, u=u, k=degree, s=smoothing_factor)

    # total arc length of B-spline
    # we compute the arc length by numerically integrating the derivative of the spline
    def spline_derivative_magnitude(_u, _tck):
        # First derivative of B-spline at _u
        dx_du, dy_du = splev(_u, _tck, der=1)
        # Return magnitude of derivative
        return np.sqrt(dx_du**2 + dy_du**2)

    # integrate over the interval [0, 1]
    arc_length_spline, _ = quad(spline_derivative_magnitude, 0, 1, args=(tck,))

    # dense sampling distance of arc length
    ds = 0.1
    # number of evaluation points
    num_eval_points = (np.ceil(arc_length_spline / ds)).astype(int)

    # evaluate spline at discrete points
    u_new = np.linspace(u.min(), u.max(), num_eval_points)
    x_new, y_new = splev(u_new, tck, der=0)

    # create polyline
    new_polyline = np.array([x_new, y_new]).transpose()

    # if desired, resample polyline with coarser distance
    if step is not None:
        new_polyline = resample_polyline(new_polyline, step)

    return new_polyline


def smooth_polyline_rdp(polyline: np.ndarray, tol=2e-5) -> np.ndarray:
    """
    Smooths a polyline using Ramer-Douglas-Peucker algorithm.
    RDP is a point reduction algorithm to simplify polylines
    """
    list_reduced_polyline = gpc_utils.reducePointsDP(polyline.tolist(), tol)
    return np.asarray(list_reduced_polyline)


def smooth_polyline_elastic_band(
        polyline: np.ndarray,
        max_deviation: float = 0.15,
        weight_smooth: float = 1.0,
        weight_lat_error: float = 0.001,
        solver_max_iter: int = 20000
) -> np.ndarray:
    """
    Smooths a polyline using an elastic band optimization with the QP formulation:
    - min. 1/2 * x^T * P * x + q * x
    - s.t. lower_bound <= A * x <= upper_bound

    Reference:
    ----------
    - https://autowarefoundation.github.io/autoware.universe/main/planning/autoware_path_smoother/docs/eb/
    - Reimplemented by: Kilian Northoff, Tobias Mascetta
    :param polyline: input polyline as np.ndarray
    :param max_deviation: constraint for max lateral deviation
    :param weight_smooth: weight for smothing
    :param weight_lat_error: weight for lateral error
    :param solver_max_iter: max iterations solver
    :return: smoothed polyline as np.ndarray
    """
    # Pre-process for optimization stability: use Chaikins to smooth out jerky parts
    iter_cnt = 0
    while iter_cnt < 10:
        polyline = resample_polyline(polyline, 2.0)
        polyline = chaikins_corner_cutting(polyline, refinements=3)
        iter_cnt += 1

    #  Pre-process for optimization stability: downsample coarsely
    delta_s = 1.0
    polyline = resample_polyline(polyline, delta_s)

    # init vectors and matrices of QP problem
    n = polyline.shape[0]   # num points
    q = np.zeros(2 * n)
    P = np.zeros((2 * n, 2 * n))
    A = sparse.identity(n)
    x_vec = np.concatenate((polyline[:, 0], polyline[:, 1]))

    # orientation matrix
    theta_vec = compute_orientation_from_polyline(polyline)
    sin_theta = np.sin(theta_vec)
    cos_theta = np.cos(theta_vec)
    theta_mat = np.zeros((n, 2 * n))
    fill_values = [(i, i, -sin_theta[i]) for i in range(n)] + \
                  [(i, i + n, cos_theta[i]) for i in range(n)]
    for val in fill_values:
        theta_mat[val[0], val[1]] = val[2]

    # P matrix
    for offset in [0, n]:
        P[offset, offset + 0] = 1
        P[offset, offset + 1] = -2
        P[offset, offset + 2] = 1
        P[offset + 1, offset + 0] = -2
        P[offset + 1, offset + 1] = 5
        P[offset + 1, offset + 2] = -4
        P[offset + 1, offset + 3] = 1
        P[offset + n - 1, offset + n - 1] = 1
        P[offset + n - 1, offset + n - 2] = -2
        P[offset + n - 1, offset + n - 3] = 1
        P[offset + n - 2, offset + n - 1] = -2
        P[offset + n - 2, offset + n - 2] = 5
        P[offset + n - 2, offset + n - 3] = -4
        P[offset + n - 2, offset + n - 4] = 1
    for k in range(2, n - 2):
        for offset in [0, n]:
            P[offset + k, offset + k - 2] = 1
            P[offset + k, offset + k - 1] = -4
            P[offset + k, offset + k] = 6
            P[offset + k, offset + k + 1] = -4
            P[offset + k, offset + k + 2] = 1

    # compute combined P matrix (smooth and lat error)
    P_smooth = weight_smooth * P
    theta_P_mat = np.dot(theta_mat, P_smooth)
    P_smooth = np.dot(theta_P_mat, theta_mat.transpose())
    P_lat_error = weight_lat_error * np.identity(n)
    P_comb = P_smooth + P_lat_error

    # compute q vector
    q = np.dot(theta_P_mat, x_vec)

    # compute bounds
    lb = -max_deviation * np.ones(n)
    ub = max_deviation * np.ones(n)
    lb[0] = 0.0
    ub[0] = 0.0

    # setup and solve
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(P_comb),
        q=q,
        A=A,
        l=lb,
        u=ub,
        max_iter=solver_max_iter,
        eps_rel=1.0e-4,
        eps_abs=1.0e-8,
        verbose=False,
    )
    res = solver.solve()

    # create polyline from lat offset
    lat_offset = res.x
    x_coords_new = list()
    y_coords_new = list()
    for i in range(n):
        x_coords_new.append(x_vec[i] + sin_theta[i] * lat_offset[i])
        y_coords_new.append(x_vec[i + n] + cos_theta[i] * lat_offset[i])

    # create new polyline from optimized coords
    new_polyline = np.array([x_coords_new, y_coords_new]).transpose()

    return new_polyline


class Interpolator:
    """
    Factory class for creating 1d interpolation functions.

    Supported interpolation types:
    - "linear": Piecewise linear interpolation
    - "cubic": Cubic spline interpolation
    - "akima": Akima spline interpolation
    """
    _dict_interp_functions: Dict[str, Callable] = {
        "linear": interp1d,
        "cubic": CubicSpline,
        "akima": Akima1DInterpolator
    }

    @classmethod
    def get_function(cls,
                     x: np.ndarray,
                     y: np.ndarray,
                     interp_type: str = "linear",
                     **kwargs):
        """
        Returns an interpolation function for the given interpolation type
        :param x: 1d array of independent variable
        :param y: 1d array of dependent variable
        :param interp_type: string for the type of interpolation
        """
        func: Optional[Callable] = cls._dict_interp_functions.get(interp_type)

        if func is None:
            raise KeyError(f"Unsupported interpolation type: {interp_type}. "
                           f"Supported types: {list(cls._dict_interp_functions.keys())}")
        return func(x, y, **kwargs)


def resample_polyline_python(polyline: np.ndarray, step: float = 2.0) -> np.ndarray:
    """
    Resamples point with equidistant spacing. Python implementation of the pycrccosy.Util.resample_polyline()

    :param polyline: polyline with 2D points
    :param step: sampling interval
    :return: resampled polyline
    """
    if len(polyline) < 2:
        return np.array(polyline)
    new_polyline = [polyline[0]]
    current_position = step
    current_length = np.linalg.norm(polyline[0] - polyline[1])
    current_idx = 0
    while current_idx < len(polyline) - 1:
        if current_position >= current_length:
            current_position = current_position - current_length
            current_idx += 1
            if current_idx > len(polyline) - 2:
                break
            current_length = np.linalg.norm(polyline[current_idx + 1]
                                            - polyline[current_idx])
        else:
            rel = current_position / current_length
            new_polyline.append((1 - rel) * polyline[current_idx] +
                                rel * polyline[current_idx + 1])
            current_position += step
    if np.linalg.norm(new_polyline[-1] - polyline[-1]) >= 1e-6:
        new_polyline.append(polyline[-1])
    return np.array(new_polyline)


def resample_polyline_cpp(polyline: np.ndarray, step: float = 2.0) -> np.ndarray:
    """
    Resamples point with equidistant spacing.

    :param polyline: polyline with 2D points
    :param step: sampling interval
    :return: resampled polyline
    """
    new_polyline = pycrccosy.Util.resample_polyline(polyline, step)
    return np.array(new_polyline)


def resample_polyline_with_length_check(polyline, length_to_check: float = 2.0):
    """
    Resamples point with length check.

    :param length_to_check: length to be checked
    :param polyline: polyline with 2D points
    :return: resampled polyline
    """
    length = np.linalg.norm(polyline[-1] - polyline[0])
    if length > length_to_check:
        polyline = resample_polyline(polyline, 1.0)
    else:
        polyline = resample_polyline(polyline, length / 10.0)

    return polyline


def resample_polyline(
        polyline: np.ndarray,
        step: float = 2.0,
        interpolation_type: str = "linear"
) -> np.ndarray:
    """
    Resamples polyline with the given resampling step (i.e., arc length).

    :param polyline: input polyline
    :param step: resampling distance of arc length
    :param interpolation_type: method for interpolation (see class Interpolator for details)
    :return: resampled polyline
    """
    # get pathlength s
    s = compute_pathlength_from_polyline(polyline)

    # get interpolation functions
    x = polyline[:, 0]
    y = polyline[:, 1]
    interp_x = Interpolator.get_function(s, x, interpolation_type)
    interp_y = Interpolator.get_function(s, y, interpolation_type)

    # resampling
    num_samples = np.ceil(s[-1] / step + 1).astype(int)
    s_resampled = np.linspace(start=0, stop=s[-1], num=num_samples)

    # resampled polyline
    new_polyline = np.column_stack(
        (
            interp_x(s_resampled),
            interp_y(s_resampled)
        )
    )
    return new_polyline


def resample_polyline_adaptive(
        polyline: np.ndarray,
        min_ds: float = 0.5,
        max_ds: float = 5.0,
        interpolation_type: str = "linear",
        factor: Optional[float] = None
) -> np.ndarray:
    """
    Adaptively resamples a given polyline according to the curvature.
    This function produces a polyline with non-uniform sampling distance. More samples are placed in parts with higher
    curvature.

    :param polyline: original polyline (equidistantly sampled)
    :param min_ds: minimum step for sampling
    :param max_ds: maximum step for sampling
    :param interpolation_type: method for interpolation (see class Interpolator for details)
    :param factor: proportionality factor between arclength distance and curvature radius at a point
    """
    # path length array of polyline
    pathlength = compute_pathlength_from_polyline(polyline)
    # curvature array of polyline
    curvature_array = compute_curvature_from_polyline_python(polyline)
    max_curvature = np.max(curvature_array)

    # proportionality factor between arc length distance and curvature radius at a point (if not given)
    _alpha = 1/(min_ds * max_curvature)
    alpha = factor if factor is not None else _alpha

    # init lists
    x = polyline[:, 0]
    y = polyline[:, 1]
    _x_new = []
    _y_new = []
    _wp_new = []

    # first point equals the first point of the original polyline
    _x_new.append(x[0])
    _y_new.append(y[0])
    _wp_new.append(pathlength[0])

    # interpolation in x and y
    interp_x = Interpolator.get_function(pathlength, x, interp_type=interpolation_type)
    interp_y = Interpolator.get_function(pathlength, y, interp_type=interpolation_type)

    ds = max_ds
    for i in range(len(x)):
        if ds < pathlength[i] - _wp_new[-1] or i == (len(x) - 1):
            if i == (len(x) - 1):
                # if last point is reached, just add last point of original polyline
                ds = pathlength[-1] - _wp_new[-1]
            s = _wp_new[-1] + ds

            # interpolate for new x, y, curvature at pathlength s
            _x_new.append(float(interp_x(s)))
            _y_new.append(float(interp_y(s)))
            _wp_new.append(s)
            # reset
            ds = max_ds

        curvature_radius = 1 / abs(curvature_array[i])
        ds = min(ds,  1 / alpha * curvature_radius)
        ds = max(ds, min_ds)

    resampled_polyline = np.column_stack((_x_new, _y_new))
    return resampled_polyline
