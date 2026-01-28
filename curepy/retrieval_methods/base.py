"""Base class for retrieval methods"""

from abc import ABC, abstractmethod
from curepy.container.measurement_function import MeasurementFunction
from curepy.container.ancillary_parameter import AncillaryParameter
from curepy.container.prior import Prior
from curepy.container.measurement import Measurement

class BaseRetrieval(ABC):
    """Base retrieval object"""
    
    def __init__(
        self,
    parallel_cores = 1,
    ):
        self.parallel_cores = parallel_cores
    
    @abstractmethod
    def run_retrieval(self):
        pass
    
if __name__ == "__main__":
    pass