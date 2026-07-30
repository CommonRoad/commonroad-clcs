# standard imports
from typing import Any, Union, Optional, Dict
import pathlib
import inspect
import enum
import dataclasses
from dataclasses import dataclass, field

# third party
from omegaconf import OmegaConf


@enum.unique
class ProcessingOption(enum.Enum):
    """Enum for different pre-processing options for reference path."""

    NONE = "none"
    """No pre-processing"""

    RESAMPLE = "resample"
    """Only resample the reference path with the desired sampling options."""

    CURVE_SUBDIVISION = "curve_subdivision"
    """Smooth the reference path using iterative curve subdivision."""

    SPLINE_SMOOTHING = "spline_smoothing"
    """Smooth the reference path using smoothing splines"""

    ELASTIC_BAND = "elastic_band"
    """Smooth the reference path using elastic band optimization"""

    MAP_COVERAGE = "map_coverage"
    """Modify the reference path to avoid ambiguous curvilinear projections for a given part of the map"""


@enum.unique
class ResamplingOption(enum.Enum):
    """Enum for resampling option of the pre-processed reference path"""

    FIXED = "fixed"
    """Resample polyline with segments of fixed length"""

    ADAPTIVE = "adaptive"
    """Adjust segment length for resampling to curvature; more points are placed in parts with higher curvature"""


def _dict_to_params(dict_params: Dict[str, Any], cls: Any) -> Any:
    """
    Converts dictionary to parameter class.

    :param dict_params: Dictionary containing parameters.
    :param cls: Parameter dataclass to which dictionary should be converted to.
    :return: Parameter class.
    """
    fields = dataclasses.fields(cls)
    cls_map = {f.name: f.type for f in fields}
    kwargs = {}
    for k, v in cls_map.items():
        if k not in dict_params:
            continue
        if inspect.isclass(v) and issubclass(v, BaseParam):
            kwargs[k] = _dict_to_params(dict_params[k], cls_map[k])
        else:
            kwargs[k] = dict_params[k]
    return cls(**kwargs)


@dataclass
class BaseParam:
    """Base configuration class"""

    def __getitem__(self, item: str) -> Any:
        """
        Getter for base parameter value.
        """
        try:
            value = self.__getattribute__(item)
        except AttributeError as e:
            raise KeyError(f"{item} is not a parameter of {self.__class__.__name__}") from e
        return value

    def __setitem__(self, key: str, value: Any):
        """
        Setter for item.
        """
        try:
            self.__setattr__(key, value)
        except AttributeError as e:
            raise KeyError(f"{key} is not a parameter of {self.__class__.__name__}") from e

    @classmethod
    def load(cls, file_path: Union[pathlib.Path, str], scenario_name: Optional[str] = None) -> 'BaseParam':
        """
        Loads parameters from a config yaml file and returns the Configuration class.

        :param file_path: Path to yaml file containing config parameters.
        :param scenario_name: If provided, scenario and planning problem are loaded from a CR scenario XML file.
        :return: Base parameter class.
        """
        file_path = pathlib.Path(file_path)
        assert file_path.suffix == ".yaml", f"File type {file_path.suffix} is unsupported! Please use .yaml!"
        loaded_yaml = OmegaConf.load(file_path)
        params = _dict_to_params(OmegaConf.to_object(loaded_yaml), cls)
        if scenario_name:
            params.general.set_path_scenario(scenario_name)
        return params

    def save(self, file_path: Union[pathlib.Path, str]):
        """
        Save config parameters to yaml file.

        :param file_path: Path where yaml file should be stored.
        """
        # Avoid saving private attributes
        dict_cfg = dataclasses.asdict(
            self, dict_factory=lambda items: {key: val for key, val in items if not key.startswith("_")}
        )
        OmegaConf.save(OmegaConf.create(dict_cfg), file_path, resolve=True)


@dataclass
class ResamplingParams(BaseParam):
    """Configuration parameters for resampling"""
    # resampling option
    option: ResamplingOption = ResamplingOption.FIXED

    # interpolation type
    # Options: "linear", "cubic", "akima"
    interpolation_type: str = "cubic"

    # fixed resampling step
    fixed_step: float = 1.0

    # adaptive resampling
    min_step: float = 0.4
    max_step: float = 1.0


@dataclass
class SplineParams(BaseParam):
    """Configuration parameters for smoothing splines using Scipy"""
    # spline degree
    degree_spline: int = 3

    # smoothing factor to adjust smoothness and fit to points
    # s=0.0 corresponds to spline interpolation
    smoothing_factor: float = 25.0


@dataclass
class SubdivisionParams(BaseParam):
    """Configuration parameters for curve subdivision algorithm"""
    # degree curve subdivision with Lane-Riesenfeld algorithm
    # 1: converges to quadratic B-Spline (i.e., Chaikins algorithm)
    # 2: converges to cubic B-Spline
    degree: int = 2

    # number of curve subdivision refinements per step
    num_refinements: int = 3

    # coarse resampling step for iterative curvature reduction
    coarse_resampling_step: float = 2.0

    # desired maximum curvature limit
    max_curvature: float = 0.135

    # maximum lateral deviation from original path
    max_deviation: float = 3.0

    # maximum number of iterations for subdivision
    max_iterations: int = 300


@dataclass
class ElasticBandParams(BaseParam):
    """Configuration parameters for smoothing via elastic band optimization"""
    # constraint for max lateral deviation of points
    max_deviation: float = 0.15

    # coarse resampling step of the input path before QP optimization
    input_resampling_step: float = 1.0

    # weights for optimization
    weight_smooth: float = 1.0
    weight_lat_error: float = 0.001

    # OSQP solver params
    max_iteration: int = 20000


@dataclass
class CLCSParams(BaseParam):
    """Configuration parameters for CLCS."""

    # processing option for reference path
    processing_option: ProcessingOption = ProcessingOption.RESAMPLE

    # ===== parameters for C++ backend
    # default lateral projection domain limit
    default_proj_domain_limit: float = 40.0
    # C++ logging level | Options: "off", "critical", "err", "warn", "info", "debug", "trace"
    logging_level: str = "off"

    # ===== parameters only for development purposes
    # tolerance projection domain computation
    eps: float = 0.1
    # tolerance extension of reference path at start and end to avoid projection errors
    eps2: float = 1e-2
    # method for projection domain computation
    method: int = 1

    # ===== parameter sub classes
    resampling: ResamplingParams = field(default_factory=ResamplingParams)
    spline: SplineParams = field(default_factory=SplineParams)
    subdivision: SubdivisionParams = field(default_factory=SubdivisionParams)
    elastic_band: ElasticBandParams = field(default_factory=ElasticBandParams)
