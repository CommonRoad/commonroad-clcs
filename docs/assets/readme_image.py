import os
from matplotlib import pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner

from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption,
    ResamplingOption
)
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs import  pycrccosy
from commonroad_clcs.helper.visualization import (
    plot_scenario_and_clcs,
)


# ***********************
# Open scenario and pp
# ***********************
scenario_name = "ZAM_Tjunction-1_42_T-1.xml"

file_path = os.path.join(os.getcwd(), "../../tutorials", scenario_name)
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]


# ********************************
# Plan Routes and Reference Path
# ********************************
route_planner = RoutePlanner(scenario.lanelet_network, planning_problem)
routes = route_planner.plan_routes()
ref_path_planner = ReferencePathPlanner(
    lanelet_network=scenario.lanelet_network,
    planning_problem=planning_problem,
    routes=routes)
ref_path = (
    ref_path_planner.plan_shortest_reference_path(
        retrieve_shortest=True,
        consider_least_lance_changes=True).reference_path
)


# *******************************************
# Reference Path Processing
# *******************************************
# set params
params = CLCSParams(processing_option=ProcessingOption.CURVE_SUBDIVISION)
# subdivision
params.subdivision.degree = 2
params.subdivision.num_refinements = 3
params.subdivision.coarse_resampling_step = 1.0
params.subdivision.max_curvature = 0.125
params.subdivision.max_deviation = 10.0
# resampling
params.resampling.option = ResamplingOption.ADAPTIVE
params.resampling.min_step = 0.4
params.resampling.max_step = 2.0
# init ref path processor
ref_path_processor = ProcessorFactory.create_processor(params)

ref_path = ref_path_processor(ref_path)


# *******************************************
# CCosy
# *******************************************
ccosy_settings = {
    "default_limit": 40.0,
    "eps": 0.1,
    "eps2": 1e-2,
    "method": 2
}

curvilinear_cosy = pycrccosy.CurvilinearCoordinateSystem(ref_path,
                                                         ccosy_settings["default_limit"],
                                                         ccosy_settings["eps"],
                                                         ccosy_settings["eps2"],
                                                         ccosy_settings["method"])


# ***************************
# Visualize
# ***************************
plot_limits = [-6, 60, -10, 55]
scenario.remove_obstacle(scenario.obstacle_by_id(7))

# save path
savepath = os.path.join(os.getcwd(), "output")

seg_list = curvilinear_cosy.get_segment_list()

for ts in range(10, 140):
    print(ts)
    rnd = MPRenderer(figsize=(7, 10), plot_limits=plot_limits)
    plot_scenario_and_clcs(
        scenario,
        curvilinear_cosy,
        renderer=rnd,
        proj_domain_plot=None,
        time_step = ts
    )

    for obs_id in [1, 4, 5]:
        obs = scenario.obstacle_by_id(obs_id)
        obs_pos_cart = obs.prediction.trajectory.state_at_time_step(ts).position
        obs_pos_curv = curvilinear_cosy.convert_to_curvilinear_coords(obs_pos_cart[0], obs_pos_cart[1])

        obs_pos_projected = curvilinear_cosy.convert_to_cartesian_coords(obs_pos_curv[0], 0.0)

        rnd.ax.plot(
            [obs_pos_cart[0], obs_pos_projected[0]],
            [obs_pos_cart[1], obs_pos_projected[1]],
            zorder=100,
            linewidth=2,
            marker='.',
            markersize=8,
            color="#e37222"
        )
        plt.axis('off')

    if ts < 100:
        save_file_name = os.path.join(savepath, f"0{ts}.svg")
    else:
        save_file_name = os.path.join(savepath, f"{ts}.svg")
    plt.savefig(save_file_name, format="svg", bbox_inches="tight", transparent=False)
