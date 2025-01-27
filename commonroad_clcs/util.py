# standard imports
import math
from typing import List, Optional, Tuple

# third party
import numpy as np

# commonroad-clcs
import commonroad_clcs.pycrccosy as pycrccosy
from commonroad_clcs.helper.interpolation import Interpolator


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
    new_polyline = pycrccosy.Util.lane_riesenfeld_subdivision(polyline, degree, refinements)
    return np.array(new_polyline)


def compute_pathlength_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the path length of a given polyline

    :param polyline: polyline with 2D points
    :return: path length of the polyline
    """
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
    curvature = pycrccosy.Util.compute_curvature(polyline, digits)
    return curvature


def compute_curvature_from_polyline_python(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvatures along a given polyline

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Curvature array of the polyline for each coordinate [1/rad]
    """
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

    # initialize for first point
    idx = 0
    curvature_radius = 1 / abs(curvature_array[idx])
    ds = min(max_ds,  1 / alpha * curvature_radius)
    ds = max(ds, min_ds)

    while idx < len(x) - 1:
        if _wp_new[-1] > pathlength[idx + 1]:
            # next idx of original polyline
            idx += 1
            # compute current ds based on local curvature of original polyline at current idx
            curvature_radius = 1 / abs(curvature_array[idx])
            ds = min(max_ds,  1 / alpha * curvature_radius)
            ds = max(ds, min_ds)
        else:
            # new s coordinate
            s = _wp_new[-1] + ds
            if s <= pathlength[-1]:
                # add new s coordinate
                _wp_new.append(s)
            else:
                # reached end of path: add s coordinate of last point and break
                _wp_new.append(pathlength[-1])
                break

    # interpolate x and y values at resampled s positions
    _wp_new = np.array(_wp_new)
    _x_new = interp_x(_wp_new)
    _y_new = interp_y(_wp_new)

    resampled_polyline = np.column_stack((_x_new, _y_new))
    return resampled_polyline


def reducePointsDP(cont, tol):
    """
    Implementation taken from the Polygon3 package, which is a Python package built around the
    General Polygon Clipper (GPC) Library
    Source: https://github.com/jraedler/Polygon3/blob/master/Polygon/Utils.py#L188
    ----------------------------------------------------------------------------------------------

    Remove points of the contour 'cont' using the Douglas-Peucker algorithm. The
    value of tol sets the maximum allowed difference between the contours. This
    (slightly changed) code was written by Schuyler Erle and put into public
    domain. It uses an iterative approach that may need some time to complete,
    but will give better results than reducePoints().

    :param cont: list of points (contour)
    :param tol: allowed difference between original and new contour
    :return new list of points
    """
    anchor  = 0
    floater = len(cont) - 1
    stack   = []
    keep    = set()
    stack.append((anchor, floater))
    while stack:
        anchor, floater = stack.pop()
        # initialize line segment
        # if cont[floater] != cont[anchor]:
        if cont[floater][0] != cont[anchor][0] or cont[floater][1] != cont[anchor][1]:
            anchorX = float(cont[floater][0] - cont[anchor][0])
            anchorY = float(cont[floater][1] - cont[anchor][1])
            seg_len = math.sqrt(anchorX ** 2 + anchorY ** 2)
            # get the unit vector
            anchorX /= seg_len
            anchorY /= seg_len
        else:
            anchorX = anchorY = seg_len = 0.0
        # inner loop:
        max_dist = 0.0
        farthest = anchor + 1
        for i in range(anchor + 1, floater):
            dist_to_seg = 0.0
            # compare to anchor
            vecX = float(cont[i][0] - cont[anchor][0])
            vecY = float(cont[i][1] - cont[anchor][1])
            seg_len = math.sqrt( vecX ** 2 + vecY ** 2 )
            # dot product:
            proj = vecX * anchorX + vecY * anchorY
            if proj < 0.0:
                dist_to_seg = seg_len
            else:
                # compare to floater
                vecX = float(cont[i][0] - cont[floater][0])
                vecY = float(cont[i][1] - cont[floater][1])
                seg_len = math.sqrt( vecX ** 2 + vecY ** 2 )
                # dot product:
                proj = vecX * (-anchorX) + vecY * (-anchorY)
                if proj < 0.0:
                    dist_to_seg = seg_len
                else:  # calculate perpendicular distance to line (pythagorean theorem):
                    dist_to_seg = math.sqrt(abs(seg_len ** 2 - proj ** 2))
                if max_dist < dist_to_seg:
                    max_dist = dist_to_seg
                    farthest = i
        if max_dist <= tol: # use line segment
            keep.add(anchor)
            keep.add(floater)
        else:
            stack.append((anchor, farthest))
            stack.append((farthest, floater))
    keep = list(keep)
    keep.sort()
    return [cont[i] for i in keep]