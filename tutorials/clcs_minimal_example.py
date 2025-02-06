import os

from commonroad.common.file_reader import CommonRoadFileReader

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner

from matplotlib import pyplot as plt
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.ref_path_processing.factory import ProcessorFactory
from commonroad_clcs.helper.visualization import plot_scenario_and_clcs
from commonroad_clcs.config import (
    CLCSParams,
    ProcessingOption,
    ResamplingOption
)


# *************** Load scenario
# scenario name
scenario_name = 'USA_Peach-2_1_T-1.xml'

# load scenario and planning problem
file_path = os.path.join(os.getcwd(), scenario_name)
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]


# *************** Plan initial reference path
# plan routes and reference path
routes = RoutePlanner(scenario.lanelet_network, planning_problem).plan_routes()
ref_path = ReferencePathPlanner(
    lanelet_network=scenario.lanelet_network,
    planning_problem=planning_problem,
    routes=routes
).plan_shortest_reference_path().reference_path


# *************** Pre-process reference path
# initialize CLCS params
params = CLCSParams()

# pre-process reference path: Here we smooth the path using curve subdivision
params.processing_option = ProcessingOption.CURVE_SUBDIVISION
params.subdivision.max_curvature = 0.15
params.resampling.option = ResamplingOption.ADAPTIVE
ref_path_processor = ProcessorFactory.create_processor(params)
ref_path = ref_path_processor(ref_path)


# *************** Construct CLCS
curvilinear_cosy = CurvilinearCoordinateSystem(
    reference_path=ref_path,
    params=params,
    preprocess_path=False
)


# *************** Transform exemplary point
cartesian_pt = [-20.0, 68.0]
curvilinear_pt = curvilinear_cosy.convert_to_curvilinear_coords(cartesian_pt[0], cartesian_pt[1])



# *************** Visualize
# plot scenario with
plot_scenario_and_clcs(
    scenario,
    curvilinear_cosy,
    proj_domain_plot="full"
)
plt.show()