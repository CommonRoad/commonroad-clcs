# standard imports
import math
from typing import List, Optional, Tuple

# third party
import numpy as np

# commonroad
from commonroad.scenario.lanelet import LaneletNetwork

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


def remove_duplicated_points_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """Removes overlapping points from the input polyline"""
    _, idx = np.unique(polyline, axis=0, return_index=True)
    polyline = polyline[np.sort(idx)]

    return polyline


def append_lanelet_centerpoints(
        reference_path: np.ndarray,
        resample_step: float,
        max_distance: float,
        where: str,
        lanelet_network: LaneletNetwork
) -> np.ndarray:
    """
    Extends a reference path up to a given maximum distance by appending the centerpoints of successor/predecessor
    lanelets.

    :param reference_path: input reference path
    :param resample_step: desired resampling step
    :param max_distance: maximum distance for appending center points
    :param where: direction to append centerpoints to lanelet
                - "front": appends centerpoints of successor lanelets towards the front (starting at the last point)
                - "back": appends centerpoints of predecessor lanelets towards the back (starting at the first point)
    :param lanelet_network: Lanelet network from CommonRoad scenario
    :return: new reference path
    """
    # check validity
    assert where in ["front", "back"], f"Invalid argument for where={where}"

    # current length and target length
    curr_length = compute_polyline_length(reference_path)
    max_length = curr_length + max_distance

    # determine starting lanelet (lanelet of first or last point of reference path)
    start_pt = reference_path[-1] if where == "front" else reference_path[0]
    list_start_lanelet_ids = lanelet_network.find_lanelet_by_position([start_pt])

    if list_start_lanelet_ids and list_start_lanelet_ids[0]:
        # take first lanelet in list
        start_lanelet = lanelet_network.find_lanelet_by_id(list_start_lanelet_ids[0][0])

        # get center vertices
        center_vertices  = resample_polyline(
            start_lanelet.center_vertices,
            step=resample_step
        )

        # find the closest point to start point
        idx_closest_pt = np.argmin(
            np.linalg.norm(center_vertices - start_pt, axis=1)
        )

        # remove all points including the closest point
        center_vertices = (
            center_vertices[(idx_closest_pt + 1):] if where == "front"
            else center_vertices[:(max(0, idx_closest_pt - 1))]
        )

        # add center vertices to reference path
        if center_vertices.size > 0:
            reference_path = (
                np.concatenate((reference_path, center_vertices), axis=0) if where == "front"
                else np.concatenate((center_vertices, reference_path), axis=0)
            )

        # compute current length
        curr_length = compute_polyline_length(reference_path)

        # get next (successor or predecessor) lanelet
        if where == "front":
            next_lanelet = (
                lanelet_network.find_lanelet_by_id(start_lanelet.successor[0]) if start_lanelet.successor
                else None
            )
        else:
            next_lanelet = (
                lanelet_network.find_lanelet_by_id(start_lanelet.predecessor[0]) if start_lanelet.predecessor
                else None
            )

        # iterate over next lanelets
        while next_lanelet is not None and curr_length < max_length:
            # get center vertices
            center_vertices = resample_polyline(
                next_lanelet.center_vertices,
                step=resample_step
            )

            # check length
            diff_length = max_length - curr_length

            # path length of center vertices next lanelet
            path_length_next = compute_pathlength_from_polyline(center_vertices)

            # case extension to front
            if where == "front":
                # clip center vertices if length exceeds diff length
                clip_idx = np.argmax(path_length_next > diff_length)
                center_vertices = center_vertices[:(max(0, clip_idx - 1))]
                # append center vertices to reference path
                reference_path = np.concatenate((reference_path, center_vertices), axis=0)
                # get next lanelet
                next_lanelet = (
                    lanelet_network.find_lanelet_by_id(next_lanelet.successor[0]) if next_lanelet.successor
                    else None
                )

            # case extension to back
            elif where == "back":
                # clip center vertices if length exceeds diff length
                clip_idx = np.argmax(path_length_next > path_length_next[-1] - diff_length)
                center_vertices = center_vertices[(max(0, clip_idx - 1)):]
                # append center vertices to reference path
                reference_path = np.concatenate((center_vertices, reference_path), axis=0)
                # get next lanelet
                next_lanelet = (
                    lanelet_network.find_lanelet_by_id(next_lanelet.predecessor[0]) if next_lanelet.predecessor
                    else None
                )

            # update current length
            curr_length = compute_polyline_length(reference_path)

    return reference_path


def extend_reference_path(
        reference_path: np.ndarray,
        resample_step: float,
        extend_front_length: float,
        extend_back_length: float,
        lanelet_network: Optional[LaneletNetwork] = None
) -> np.ndarray:
    """
    Extends a reference path at the front and back by a given length.
    Note: The extended reference path is resampled uniformly and not necessarily C2 continuous.
          Post-processing / smoothing should be done afterward.

    :param reference_path: Original reference path
    :param resample_step: desired resampling step of extended polyline
    :param extend_front_length: length for front extension
    :param extend_back_length: length for back extension
    :param lanelet_network: Lanelet network from CommonRoad scenario to extend using lanelet centerpoints (optional)
    :return: extended reference path
    """
    # resample polyline initially
    reference_path = resample_polyline(reference_path, step=resample_step)

    # ----------- front extension
    # target reference path length
    target_length = compute_polyline_length(reference_path) + extend_front_length

    # append lanelet centerpoints for front extension
    reference_path = append_lanelet_centerpoints(
        reference_path,
        resample_step=resample_step,
        max_distance=extend_front_length,
        where="front",
        lanelet_network=lanelet_network
    )

    # update current length
    curr_length = compute_polyline_length(reference_path)

    # linear extrapolation in front if target length is not yet reached
    if curr_length < target_length:
        reference_path = extrapolate_polyline(
            polyline=reference_path,
            distance=target_length - curr_length,
            where="front",
            resample_step=resample_step
        )

    # ----------- back extension
    # target reference path length
    target_length = compute_polyline_length(reference_path) + extend_back_length

    # append lanelet centerpoints for back extension
    reference_path = append_lanelet_centerpoints(
        reference_path,
        resample_step=resample_step,
        max_distance=extend_back_length,
        where="back",
        lanelet_network=lanelet_network
    )

    # update current length
    curr_length = compute_polyline_length(reference_path)

    # linear extrapolation behind if target length is not yet reached
    if curr_length < target_length:
        reference_path = extrapolate_polyline(
            polyline=reference_path,
            distance=target_length - curr_length,
            where="back",
            resample_step=resample_step
        )

    # remove potential duplicate points
    reference_path = remove_duplicated_points_from_polyline(reference_path)
    return reference_path


def extrapolate_polyline(
        polyline: np.ndarray,
        distance: float,
        where: str,
        resample_step: float = 2.0,
) -> np.ndarray:
    """
    Function to extrapolate the end of a polyline linearly by a given distance.
    :param polyline: input polyline
    :param distance: Distance to extend polyline
    :param resample_step: interval for resampling
    :param where: direction to extrapolate the polyline
                - "front": extrapolates towards the front (starting at the last point)
                - "back": extrapolates towards the back (starting at the first point)
    :return extrapolated and resampled reference path
    """
    # check input polyline
    assert (isinstance(polyline, np.ndarray) and
            polyline.ndim == 2 and
            len(polyline[:,0]) > 2), 'Input polyline mus be a numpy array with ndim ==2 and more than 2 points'

    if where == "front":
        # last two points
        p1, p2 = polyline[-2], polyline[-1]
        # direction vector
        direction = p2 - p1
        # extrapolated point
        new_point = p2 + (direction / np.linalg.norm(direction)) * distance
        # append new point to the last point
        extended_polyline = np.vstack([polyline, new_point])

    elif where == "back":
        # first two points
        p1, p2 = polyline[0], polyline[1]
        # direction vector
        direction = p1 - p2
        # extrapolated point
        new_point = p1 + (direction / np.linalg.norm(direction)) * distance
        # append new point to the first point
        extended_polyline = np.vstack([new_point, polyline])

    else:
        raise ValueError(f"Invalid argument where={where}. Use front or back.")

    return resample_polyline(extended_polyline, step=resample_step)


def fix_polyline_vertex_ordering(
        polyline: np.ndarray,
        theta_diff_threshold: float = np.pi/2
) -> np.ndarray:
    """
    Fixes polyline with incorrect vertex ordering. Correct vertex ordering is determined based on threshold for the
    orientation difference between consecutive vertices.
    :param polyline: input polyline
    :param theta_diff_threshold: threshold for max. orientation difference between consecutive vertices
    :return fixed polyline
    """
    # we assume that the "correct" direction of the polyline is given by the first two points
    ordered_polyline = polyline[0:2, :]

    # loop over points 2:n
    for j in range(2, len(polyline)):
        ordered_polyline = np.vstack([ordered_polyline, polyline[j]])

        # iteratively push added point one place back if max orientation diff is too high
        for i in range(len(ordered_polyline) - 1, 0, - 1):
            _theta = np.unwrap(compute_orientation_from_polyline(ordered_polyline))
            _diff_theta = np.diff(_theta)

            if not all(np.abs(_diff_theta) < theta_diff_threshold):
                ordered_polyline[[i, i - 1]] = ordered_polyline[[i - 1, i]]
            else:
                break

    return ordered_polyline
