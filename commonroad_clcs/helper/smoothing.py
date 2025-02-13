# standard imports
from typing import Optional
from copy import deepcopy

# third party
import numpy as np
from scipy.integrate import quad
from scipy import sparse
from scipy.interpolate import (
    splprep,
    splev
)
import osqp
from shapely import LineString, Point

# commonroad-clcs
from commonroad_clcs.util import (
    compute_curvature_from_polyline,
    compute_orientation_from_polyline,
    resample_polyline,
    reducePointsDP,
    chaikins_corner_cutting,
    lane_riesenfeld_subdivision,
    resample_polyline_cpp
)


def smooth_polyline_subdivision(
        polyline: np.ndarray,
        degree: int,
        refinements: int = 3,
        coarse_resampling: float = 2.0,
        max_curv: Optional[float] = None,
        max_dev: Optional[float] = None,
        max_iter: int = 1000,
        verbose: bool = False,
) -> np.ndarray:
    """
    Smooths a polyline using Lane-Riesenfeld curve subdivision.
    The degree of the subdivision k has the B-spline curve of degree k+1 as it's limit curve.
    E.g., for degree k=2 the subdivision converges to a cubic B-spline with C^2 continuity.
    This function iteratively reduces the curvature and smooths the input polyline.

    :param polyline: input polyline
    :param degree: curve subdivision degree
    :param refinements: number of subdivision steps
    :param coarse_resampling: coarse resampling step in each iteration
    :param max_curv: maximum curvature for smoothing
    :param max_dev: maximum lateral deviation from ref path
    :param max_iter: maximum number of smoothing iterations (default 1000)
    :param verbose: print output
    :return: smoothed polyline as np.ndarray
    """
    new_polyline = deepcopy(polyline)

    # line string of original polyline
    original_ls = LineString(polyline)

    # get max curvature
    max_curv = max_curv if max_curv is not None else 10.0

    # iteration counter
    iter_cnt = 0

    # current maximum curvature
    curr_max_curv = np.max(compute_curvature_from_polyline(new_polyline))

    # iterative smoothing
    # Breaking conditions: curvature limit reached or max iterations reached
    while (curr_max_curv > max_curv) and (iter_cnt < max_iter):
        new_polyline = lane_riesenfeld_subdivision(new_polyline, degree, refinements)

        # get current curvature
        curv_arr = compute_curvature_from_polyline(new_polyline)

        # get max curvature
        curr_max_curv = np.max(curv_arr)

        # compute lateral deviation at point of maximum curvature
        if max_dev is not None:
            # get point of max curvature
            idx_max_curv = np.argmax(curv_arr)
            max_curv_pt = new_polyline[idx_max_curv]
            # lateral distance to original polyline
            deviation = Point(max_curv_pt).distance(original_ls)

        # resample with coarse step
        new_polyline = resample_polyline_cpp(new_polyline, coarse_resampling)

        # Optional breaking condition: max. lateral deviation is exceeded
        if max_dev is not None and deviation > max_dev:
            break

        # increase counter
        iter_cnt += 1

    # print summary
    if verbose:
        print(f"Number of iterations: {iter_cnt}")
        if max_dev is not None:
            print(f"Maximum lateral deviation: {deviation}")

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

def smooth_polyline_rdp(
        polyline: np.ndarray,
        tol: float = 2e-5,
        resample_step: Optional[float] = None
) -> np.ndarray:
    """
    Smooths a polyline using Ramer-Douglas-Peucker algorithm.
    RDP is a point reduction algorithm to simplify polylines
    :param polyline: input polyline as numpy array
    :param tol: allowed tolerance for removing points from original polyline
    :param resample_step: fixed step for resampling the modified polyline (if desired)
    """
    list_reduced_polyline = reducePointsDP(polyline.tolist(), tol)
    new_polyline = np.asarray(list_reduced_polyline)
    if resample_step:
        return resample_polyline(new_polyline, resample_step)
    else:
        return new_polyline


def smooth_polyline_elastic_band(
        polyline: np.ndarray,
        input_resampling: float = 1.0,
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
    - Source: https://autowarefoundation.github.io/autoware.universe/main/planning/autoware_path_smoother/docs/eb/
    - Reimplemented by: Kilian Northoff, Tobias Mascetta
    :param polyline: input polyline as np.ndarray
    :param input_resampling: coarse resampling of the input polyline for stability of QP solver
    :param max_deviation: constraint for max lateral deviation
    :param weight_smooth: weight for smoothing
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
    polyline = resample_polyline(polyline, input_resampling)

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
