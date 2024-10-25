# standard imports
from typing import Dict, Type

from commonroad_clcs.config import (
    ProcessingOption,
    CLCSParams
)

from .processor import IReferencePathProcessor
from .implementation import (
    NoPreProcessor,
    ResamplingProcessor,
    CurveSubdivisionProcessor,
    SplineSmoothingProcessor,
    ElasticBandProcessor,
    MapCoverageProcessor
)


class ProcessorFactory:
    """Factory class for selecting class to pre-process the reference path"""

    # map ProcessionOption to corresponding class
    _dict_processing_strategy: Dict[ProcessingOption, Type[IReferencePathProcessor]] = {
        ProcessingOption.NONE: NoPreProcessor,
        ProcessingOption.RESAMPLE: ResamplingProcessor,
        ProcessingOption.CURVE_SUBDIVISION: CurveSubdivisionProcessor,
        ProcessingOption.SPLINE_SMOOTHING: SplineSmoothingProcessor,
        ProcessingOption.ELASTIC_BAND: ElasticBandProcessor,
        ProcessingOption.MAP_COVERAGE: MapCoverageProcessor
    }

    @staticmethod
    def create_processor(params: CLCSParams) -> IReferencePathProcessor:
        """Returns processing strategy class for given processing option in params"""
        processing_option: ProcessingOption = params.processing_option
        processor_class = ProcessorFactory._dict_processing_strategy.get(processing_option)
        if processor_class:
            return processor_class(params)
        else:
            raise KeyError(f"Unknown processing option: {processing_option}")
