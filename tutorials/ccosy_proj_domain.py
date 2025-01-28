"""
Testing Script for curvilinear coordinate system functionalities
"""

from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import os
import time

# commonroad-clcs
from commonroad_clcs import pycrccosy
from commonroad_clcs.config import CLCSParams, ProcessingOption, ResamplingOption
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs.util import resample_polyline, chaikins_corner_cutting, \
    compute_curvature_from_polyline, compute_pathlength_from_polyline, \
    compute_curvature_from_polyline_python, \
    resample_polyline_adaptive, lane_riesenfeld_subdivision

from commonroad_clcs.helper.smoothing import (
    smooth_polyline_spline,
    smooth_polyline_elastic_band,
    smooth_polyline_rdp,
    smooth_polyline_subdivision
)
from commonroad_clcs.helper.visualization import plot_scenario_and_clcs, plot_segment_normal_tangent, \
    plot_reference_curvature, plot_reference_path_partitions, plot_scenario_and_pp, plot_curvilinear_projection_domain
from commonroad_clcs.helper.evaluation import (
    evaluate_ref_path_deviations,
    evaluate_ref_path_curvature_improvements
)
# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner

# commonroad-io imports
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer


# ***************************
# Load scenario
# ***************************
# load the CommonRoad scenario, note that you might have to modify the path to the CommonRoad scenario!
# scenario_name = 'USA_Peach-3_1_T-1.xml'
# scenario_name = "USA_Lanker-2_5_T-1_mod1.xml"

# scenario_name = 'USA_Peach-2_1_T-1.xml'
scenario_name = "ZAM_Tjunction-1_42_T-1.xml"
# scenario_name = "USA_Lanker-2_5_T-1_mod2.xml"

file_path = os.path.join(os.getcwd(), scenario_name)

scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
# retrieve the first planning problem in the problem set
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

# plot_scenario_and_pp(scenario, planning_problem)


# ********************************
# Set plot settings for scenario
# ********************************
# general global settings
savepath = None
savepath_curvature = None
plot_limits = None
show_plots = True
evaluate_curvature = True
_columnwidth_in = 8.87 / 2.54
_verbose = True

# adaptive sampling
_min_ds = 1.0
_max_ds = 2.0

if scenario_name == "USA_Lanker-2_5_T-1_mod2.xml":
    scenario.translate_rotate(translation=np.array([0, 0]), angle=np.pi/7)
    # planning_problem.initial_state.position[0] = -22.5
    plot_limits = [-8, 21, -8, 39.0]
    # plot limits big
    # plot_limits = [-27, 43.2, -22.9, 52.8]
    scenario_max_curv = 0.062
    savepath = os.path.join(os.getcwd(), "output/USA_Lanker-2_5_T-1_mod2.svg")
    savepath_curvature = os.path.join(os.getcwd(), "output/USA_Lanker-2_5_T-1_mod2_curvature.svg")
    # boundary_lanelet_ids = ["3450", "3604", "3517"]

if scenario_name == "USA_Peach-2_1_T-1.xml":
    plot_limits = [-40.32, -7, 26, 80]
    scenario_max_curv = 0.138
    savepath = os.path.join(os.getcwd(), "output/USA_Peach-2_1_T-1.svg")
    savepath_curvature = os.path.join(os.getcwd(), "output/USA_Peach-2_1_T-1_curvature.svg")

if scenario_name == "ZAM_Tjunction-1_42_T-1.xml":
    plot_limits = [-6, 34.105, -10, 55]
    scenario_max_curv = 0.125
    savepath = os.path.join(os.getcwd(), "output/ZAM_Tjunction-1_42_T-1.svg")
    savepath_curvature = os.path.join(os.getcwd(), "output/ZAM_Tjunction-1_42_T-1_curvature.svg")


# ********************************
# Plan Routes and Reference Path
# ********************************
print("Planning routes...")
time_start = time.time()
# instantiate a route planner with the scenario and the planning problem
route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
# plan routes, and save the routes in a route candidate holder
routes = route_planner.plan_routes()
# get initial reference path from route
ref_path_planner = ReferencePathPlanner(
    lanelet_network=scenario.lanelet_network,
    planning_problem=planning_problem,
    routes=routes)
ref_path = (
    ref_path_planner.plan_shortest_reference_path(
        retrieve_shortest=True,
        consider_least_lance_changes=True).reference_path
)

print("\tPlanning routes took: \t", time.time() - time_start)

# store original ref path for later
ref_path_orig = deepcopy(ref_path)


# *******************************************
# Pre-Processing options on reference path
# *******************************************
# apply chaikins corner cutting for fixed number of iterations and resample afterwards
corner_cutting = False
iterations_corner_cutting = 500
if corner_cutting:
    for i in range(0, iterations_corner_cutting):
        print(f"Chaikin's iteration: {i}")
        ref_path = chaikins_corner_cutting(ref_path)
        ref_path = resample_polyline(ref_path, 2.0)
        i += 1

# only resample in fixed step
resample_fixed = False
resampling_step = 1.0

if resample_fixed:
    # set params
    params = CLCSParams(processing_option=ProcessingOption.RESAMPLE)
    params.resampling.option = ResamplingOption.FIXED
    params.resampling.fixed_step = resampling_step
    # init ref path processor
    ref_path_processor = ProcessorFactory.create_processor(params)
    # process ref path
    ref_path = ref_path_processor(ref_path)


# preprocess ref path for limiting curvature via curve subdivision
max_curv_preprocess = False
degree_subdivision = 2

if max_curv_preprocess:
    # set params
    params = CLCSParams(processing_option=ProcessingOption.CURVE_SUBDIVISION)
    # subdivision
    params.subdivision.degree = degree_subdivision
    params.subdivision.num_refinements = 3
    params.subdivision.coarse_resampling_step = resampling_step
    params.subdivision.max_curvature = 0.11
    params.subdivision.max_deviation = 1.3
    # resampling
    params.resampling.option = ResamplingOption.ADAPTIVE
    params.resampling.min_step = _min_ds
    params.resampling.max_step = _max_ds
    # init ref path processor
    ref_path_processor = ProcessorFactory.create_processor(params)

    # process ref path
    print("Smoothing via curve subdivision...")
    t0 = time.perf_counter()

    ref_path = ref_path_processor(ref_path, verbose=_verbose)

    print(f"Elapsed time: {time.perf_counter() - t0}")

    if _verbose:
        print(f"Ref path length after resampling: {len(ref_path)}")


# smoothing using Douglas-Peucker algorithm
smooth_rdp = False
if smooth_rdp:
    # tol = 7e-2
    tol = 1e-1
    ref_path = smooth_polyline_rdp(polyline=ref_path, tol=tol)


# smoothing by spline interpolation scipy
smooth_spline = False
if smooth_spline:
    print("Smoothing via B-spline approximation...")
    # set params
    params = CLCSParams(processing_option=ProcessingOption.SPLINE_SMOOTHING)
    # spline
    params.spline.degree_spline = 3
    params.spline.smoothing_factor = 40.0
    # resampling
    params.resampling.option = ResamplingOption.ADAPTIVE
    params.resampling.min_step = _min_ds
    params.resampling.max_step = _max_ds

    # init ref path processor
    ref_path_processor = ProcessorFactory.create_processor(params)

    # process ref path
    ref_path = ref_path_processor(ref_path, verbose=_verbose)

    if _verbose:
        print(f"Ref path length after resampling: {len(ref_path)}")


# smoothing via elastic band optimization
smooth_eb = True
if smooth_eb:
    print("Smoothing via elastic band optimization...")
    # set params
    params = CLCSParams(processing_option=ProcessingOption.ELASTIC_BAND)
    # elastic band
    params.elastic_band.max_deviation = 0.15
    params.elastic_band.input_resampling_step = 1.0
    # resampling
    params.resampling.option = ResamplingOption.ADAPTIVE
    params.resampling.min_step = _min_ds
    params.resampling.max_step = _max_ds

    # init ref path processor
    ref_path_processor = ProcessorFactory.create_processor(params)

    # process ref path
    ref_path = ref_path_processor(ref_path, verbose=_verbose)

    if _verbose:
        print(f"Ref path length after resampling: {len(ref_path)}")


# ***************************
# Curvilinear CoSys
# ***************************
# create curvilinear CoSys ref_path after preprocessing/spline smoothing
ccosy_settings = {
    "default_limit": 40.0,
    "eps": 0.1,
    "eps2": 1e-4,
    "method": 2
}

curvilinear_cosy = pycrccosy.CurvilinearCoordinateSystem(ref_path,
                                                         ccosy_settings["default_limit"],
                                                         ccosy_settings["eps"],
                                                         ccosy_settings["eps2"],
                                                         ccosy_settings["method"])

# convert and project point
project_point = False

if project_point:
    position_cart = np.array([0, 2])
    p_curvilinear = curvilinear_cosy.convert_to_curvilinear_coords(position_cart[0], position_cart[1])
    projected_point = curvilinear_cosy.convert_to_cartesian_coords(p_curvilinear[0], 0)


# ***************************
# Visualize
# ***************************
# create renderer object
if show_plots:
    rnd = MPRenderer(figsize=(7, 10), plot_limits=plot_limits)
    plot_scenario_and_clcs(scenario, curvilinear_cosy, renderer=rnd, proj_domain_plot="full")

    # draw reference path original
    ref_path_orig = resample_polyline(ref_path_orig, resampling_step)
    rnd.ax.plot(ref_path_orig[:, 0], ref_path_orig[:, 1], zorder=99, marker=".", markersize="5", color='black')

    # draw projection of point
    if project_point:
        rnd.ax.plot([position_cart[0], projected_point[0]], [position_cart[1], projected_point[1]],
                    zorder=100, linewidth=2, marker='x', markersize=9, color='red')

    if savepath:
        plt.axis('off')
        plt.savefig(savepath, format="svg", bbox_inches="tight", transparent=False)

    if show_plots:
        plt.show()

# ***************************
# Evaluate
# ***************************
if evaluate_curvature:
    # compute curvature
    curvilinear_cosy.compute_and_set_curvature()

    # original ref path
    ref_pos_orig = compute_pathlength_from_polyline(ref_path_orig)
    ref_curv_orig = compute_curvature_from_polyline(ref_path_orig)

    # ccosy ref path
    ref_path_ccosy = np.asarray(curvilinear_cosy.reference_path())
    ref_pos_ccosy = compute_pathlength_from_polyline(ref_path_ccosy)
    ref_curv_ccosy = compute_curvature_from_polyline_python(ref_path_ccosy)

    # plot
    fig, axs = plt.subplots(2)
    fig.set_figheight(4.5/2.54)
    fig.set_figwidth(_columnwidth_in)
    plot_reference_curvature(ref_path_orig, ref_curv_orig, ref_pos_orig,
                             label="original", axs=axs, color="black", linestyle="dashed")
    plot_reference_curvature(ref_path_ccosy, ref_curv_ccosy, ref_pos_ccosy,
                             label="adapted", axs=axs, color="green")
    axs[0].legend()
    if show_plots:
        plt.show()

    ret_curv = evaluate_ref_path_curvature_improvements(ref_pos_orig, ref_curv_orig, ref_pos_ccosy, ref_curv_ccosy)
    ret = evaluate_ref_path_deviations(ref_path_orig, ref_path_ccosy, curvilinear_cosy)

    #
    # print("##################")
    # print("EVALUATION")
    # print("##################")
    # print("")
    #
    # print(f"Delta kappa average is: {ret_curv[0]}")
    # print(f"Delta kappa dot average is: {ret_curv[1]}")
    # print(f"Delta kappa max is: {ret_curv[2]}")
    # print(f"Delta kappa dot max is: {ret_curv[3]}")
    #
    # print(f"Delta s is: {ret[0]}")
    # print(f"Average delta d is: {ret[1]}")
    # print(f"Average delta theta is: {ret[2]}")

