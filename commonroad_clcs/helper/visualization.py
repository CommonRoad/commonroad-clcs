# standard imports
from typing import Optional

# third party
import matplotlib.pyplot as plt
import numpy as np

# commonroad
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.geometry.shape import Polygon

from commonroad.visualization.draw_params import (
    ShapeParams,
    LaneletNetworkParams,
    LaneletParams,
    IntersectionParams,
    TrafficLightParams
)

from commonroad.scenario.scenario import (
    Scenario,
    LaneletNetwork
)

from commonroad_clcs.pycrccosy import CurvilinearCoordinateSystem

from commonroad_clcs.util import (
    compute_pathlength_from_polyline,
    compute_curvature_from_polyline,
    get_inflection_points,
    compute_curvature_from_polyline_python
)


# dictionaries
_my_colors = {
    "proj_domain_face": "orange",
    "proj_domain_edge": "black",
    "lanelet_standard": "#e6e6e6",
    "lanelet_standard_center": "#dddddd",
    "lanelet_standard_right": "#b5b5b5",
    "lanelet_standard_left": "#b5b5b5",
    "lanelet_highlighted_face": "#c2c0c0",
    "lanelet_highlighted_center": "#3232ff",
    "normal_vectors": "black",
    "tangent_vectors": "red",
}

_ref_path_params = {
    "zorder": 100,
    "marker": ".",
    "markersize": "10",
    "linewidth": 3.0
}

_dict_proj_domain_params = {
    "facecolor": _my_colors["proj_domain_face"],
    "opacity": 0.3,
    "edgecolor": _my_colors["proj_domain_edge"],
    "linewidth": 1.5
}

_dict_lanelet_network_params = {
    "lanelet": {
        "show_label": False,
        "facecolor": _my_colors["lanelet_standard"],
        "draw_line_markings": False,
        "center_bound_color": _my_colors["lanelet_standard_center"],
        "left_bound_color": _my_colors["lanelet_standard_left"],
        "right_bound_color": _my_colors["lanelet_standard_right"]
    },
    "intersection": {
        "draw_intersections": False,
        "show_label": False,
        "draw_incoming_lanelets": True,
        "draw_successors": True
    },
    "traffic_light": {
        "draw_traffic_lights": False
    }
}

_dict_highlighted_lanelet_params = {
    "draw_stop_line": False,
    "draw_left_bound": False,
    "draw_right_bound": False,
    "draw_start_and_direction": False,
    "show_label": False,
    "draw_linewidth": 1,
    # "facecolor": '#469d89',
    "facecolor": _my_colors["lanelet_highlighted_face"],
    "center_bound_color": _my_colors["lanelet_highlighted_center"]
}


def _highlight_lanelets(renderer: MPRenderer, lanelet_network: LaneletNetwork):
    """ Highlight specific lanelets given by lanelet_network """
    lanelet_network_params = LaneletNetworkParams()
    lanelet_network_params.lanelet = LaneletParams(**_dict_highlighted_lanelet_params)
    lanelet_network_params.traffic_light = TrafficLightParams(**_dict_lanelet_network_params["traffic_light"])
    lanelet_network.draw(renderer, draw_params=lanelet_network_params)


def _make_proj_domain_polygon(clcs: CurvilinearCoordinateSystem, which: str) -> Polygon:
    """
    Creates Polygon object for the projection domain from borders stored in CLCS
    :param clcs: pycrccosy.CurvilinearCoordinateSystem object
    :param which: str indicating whether to plot "full", "left" or "right" projection domain
    """
    # ignore first and last vertex of ref path -> not included in proj domain computation
    ref_path = np.asarray(clcs.reference_path())[1:-1]
    if which == "full":
        proj_domain_border = np.asarray(clcs.projection_domain())
    elif which == "right":
        lower_proj_domain_border = np.asarray(clcs.lower_projection_domain_border())
        ref_path = np.flip(ref_path, axis=0)
        proj_domain_border = np.concatenate((lower_proj_domain_border, ref_path), axis=0)
        # make closed vertices to construct polygon object
        np.append(proj_domain_border, lower_proj_domain_border[0])
    elif which == "left":
        upper_proj_domain_border = np.asarray(clcs.upper_projection_domain_border())
        ref_path = np.flip(ref_path, axis=0)
        proj_domain_border = np.concatenate((upper_proj_domain_border, ref_path), axis=0)
        # make closed vertices to construct polygon object
        np.append(proj_domain_border, upper_proj_domain_border[0])
    else:
        raise ValueError("Please pass a valid string! Possible values: full, right, left")

    proj_domain_polygon = Polygon(proj_domain_border)
    return proj_domain_polygon


def plot_scenario_and_clcs(scenario: Scenario = None, clcs: CurvilinearCoordinateSystem = None,
                           planning_problem: Optional[PlanningProblem] = None, renderer: Optional[MPRenderer] = None,
                           show: bool = False, lanelet_network: Optional[LaneletNetwork] = None,
                           proj_domain_plot: Optional[str] = "full", proj_domain_vertices: bool = False,
                           plot_ref_path: bool = True):
    """
    Plots a scenario and given Curvilinear Coordinate system (CLCS). The CLCS is visualized with the reference path and
    the corresponding unique projection domain.
    :param scenario: Scenario object
    :param clcs: pycrccosy.CurvilinearCoordinateSystem object
    :param planning_problem: PlanningProblem object
    :param renderer: MPRenderer object
    :param show: show plot True/False
    :param lanelet_network: subset of lanelets to highlight
    :param proj_domain_plot: str, whether to plot full, left or right projection domain ("full", "left", "right")
    :param plot_ref_path bool
    """
    # create renderer if not given
    rnd = renderer if renderer is not None else MPRenderer(figsize=(7, 10))

    # draw params general
    rnd.draw_params.time_begin = 0
    rnd.draw_params.dynamic_obstacle.draw_shape = False
    rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = False

    # draw params planning problem
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False

    # shape params for proj_domain
    proj_domain_params = ShapeParams(**_dict_proj_domain_params)

    # draw params lanelet network
    lanelet_network_params = LaneletNetworkParams()
    lanelet_network_params.lanelet = LaneletParams(**_dict_lanelet_network_params["lanelet"])
    lanelet_network_params.intersection = IntersectionParams(**_dict_lanelet_network_params["intersection"])
    lanelet_network_params.traffic_light = TrafficLightParams(**_dict_lanelet_network_params["traffic_light"])
    rnd.draw_params.lanelet_network = lanelet_network_params

    # add components to renderer
    if scenario is not None:
        scenario.draw(rnd)
    if planning_problem is not None:
        planning_problem.draw(rnd)

    if proj_domain_plot is not None:
        proj_domain_polygon = _make_proj_domain_polygon(clcs, proj_domain_plot)
        proj_domain_polygon.draw(rnd, draw_params=proj_domain_params)

    if lanelet_network:
        _highlight_lanelets(rnd, lanelet_network)

    # render everything
    rnd.render()

    # draw reference path
    if plot_ref_path:
        ref_path = np.asarray(clcs.reference_path())
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], zorder=100, marker=".", color='green')

    # draw vertices of projection domain border
    if proj_domain_vertices:
        proj_domain_border = np.asarray(clcs.projection_domain())
        rnd.ax.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], zorder=100, marker=".",
                    color=_dict_proj_domain_params["edgecolor"])

    if show:
        plt.show()


def plot_segment_normal_tangent(clcs: CurvilinearCoordinateSystem, list_seg_idx: Optional[list] = None,
                                max_scale: Optional[float] = None, curvature_arr: Optional[np.ndarray] = None,
                                renderer: Optional[MPRenderer] = None, plot_normals: bool = True,
                                plot_tangents: bool = True, annotate_seg_idx: bool = True, show: bool = False):
    # create renderer if not given
    rnd = renderer if renderer is not None else MPRenderer(figsize=(7, 10))

    # get list of segments in CLCS
    list_segments = clcs.get_segment_list()
    # get reference path
    ref_path = np.array(clcs.reference_path())
    # set scaling of normals and tangents
    _scale = max_scale if max_scale is not None else 10.0

    # precision value for normal length
    _eps = 0.05

    # initialize based points, normals and tangents
    list_p1 = list()
    list_p2 = list()
    list_n1 = list()
    list_n2 = list()
    list_t1 = list()
    list_t2 = list()

    # set segment idx to plot
    _seg_idx = list_seg_idx if list_seg_idx is not None else range(len(list_segments))

    for idx in _seg_idx:
        if curvature_arr is not None:
            _scale = min(1/curvature_arr[idx] + 0.01, max_scale)
        seg = list_segments[idx]
        list_p1.append(seg.pt_1)
        list_p2.append(seg.pt_2)
        # list_n1.append(list_p1[-1] + _scale * seg.normal_segment_start)
        list_n2.append(list_p2[-1] + _scale * seg.normal_segment_end)
        list_t1.append(list_p1[-1] + _scale/2 * seg.tangent_segment_start)
        list_t2.append(list_p2[-1] + _scale/2 * seg.tangent_segment_end)
        if annotate_seg_idx:
            rnd.ax.annotate(idx, (ref_path[idx, 0], ref_path[idx, 1]), zorder=100)

    if plot_normals:
        for i, n1 in enumerate(list_n1):
            rnd.ax.plot([list_p1[i][0], n1[0]], [list_p1[i][1], n1[1]], zorder=100, color=_my_colors["normal_vectors"])
        for i, n2 in enumerate(list_n2):
            rnd.ax.plot([list_p2[i][0], n2[0]], [list_p2[i][1], n2[1]], zorder=100, color=_my_colors["normal_vectors"])
    if plot_tangents:
        for i, t1 in enumerate(list_t1):
            rnd.ax.plot([list_p1[i][0], t1[0]], [list_p1[i][1], t1[1]], zorder=100, color=_my_colors["tangent_vectors"])
        for i, t2 in enumerate(list_t2):
            rnd.ax.plot([list_p2[i][0], t2[0]], [list_p2[i][1], t2[1]], zorder=100, color=_my_colors["tangent_vectors"])

    if show:
        plt.show()

    return list_p1, list_p2, list_n1, list_n2, list_t1, list_t2


def plot_reference_path_partitions(clcs: CurvilinearCoordinateSystem, rnd: MPRenderer):
    """
    Plots inflection points of the reference path and partitions of the reference path according to inflection points
    A partition is a part of the reference path where the curvature sign doesn't change
    """
    ref_path_partitions = clcs.reference_path_partitions()
    # TUM colors: green, orange, blue
    _colors = ["#A2AD00", "#E37222", "#005293"]
    for i in range(len(ref_path_partitions)):
        color = _colors[(i % 3) - 1]
        partition = np.asarray(ref_path_partitions[i])
        rnd.ax.plot(partition[:, 0], partition[:, 1],
                    zorder=_ref_path_params["zorder"],
                    marker=_ref_path_params["marker"],
                    markersize=_ref_path_params["markersize"],
                    linewidth=_ref_path_params["linewidth"],
                    color=color)

    inflection_points, idx_inflection_pts = get_inflection_points(np.asarray(clcs.reference_path()), digits=4)

    print(f"Inflection point indices in reference path:{idx_inflection_pts}")
    rnd.ax.scatter(inflection_points[:, 0], inflection_points[:, 1], marker="x", s=50, color='black', zorder=100)

    return inflection_points, idx_inflection_pts


def plot_scenario_and_pp(scenario, planning_problem = None):
    # create renderer if not given
    rnd = MPRenderer(figsize=(7, 10))

    # draw params general
    rnd.draw_params.time_begin = 0
    rnd.draw_params.dynamic_obstacle.draw_shape = False
    # draw params lanelet network
    lanelet_network_params = LaneletNetworkParams()
    lanelet_network_params.lanelet = LaneletParams(**_dict_lanelet_network_params["lanelet"])
    lanelet_network_params.intersection = IntersectionParams(**_dict_lanelet_network_params["intersection"])
    lanelet_network_params.traffic_light = TrafficLightParams(**_dict_lanelet_network_params["traffic_light"])
    rnd.draw_params.lanelet_network = lanelet_network_params

    # render everything
    scenario.lanelet_network.draw(rnd)
    if planning_problem is not None:
        planning_problem.draw(rnd)
    rnd.render()
    plt.show()


def plot_curvilinear_projection_domain(clcs: CurvilinearCoordinateSystem):
    """Plots reference path and curvilinear projection domain for a given clcs object"""
    # manually convert Cartesian projection domain vertices to curvilinear coordinates
    cart_proj_domain = np.asarray(clcs.projection_domain())
    curv_proj_domain = list()
    for idx in range(len(cart_proj_domain)):
        point_cart = cart_proj_domain[idx]
        point_curv = clcs.convert_to_curvilinear_coords(point_cart[0], point_cart[1])
        curv_proj_domain.append(point_curv)

    curv_proj_domain = np.asarray(curv_proj_domain)

    plt.figure(figsize=(7, 10))
    # plot reference path
    ref_path = np.asarray(clcs.reference_path())
    pathlength = compute_pathlength_from_polyline(ref_path)
    plt.plot(pathlength, np.zeros(len(pathlength)), zorder=100, marker=".", color='green')
    # plot vertices of curvilinear projection domain
    plt.plot(curv_proj_domain[:, 0], curv_proj_domain[:, 1], zorder=100, marker=".", color='black')
    plt.show()
