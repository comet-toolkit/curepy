"""Base class for retrieval methods"""

from abc import ABC, abstractmethod
from curepy.container.retrieval_input import RetrievalInput
import numpy as np

class BaseRetrieval(ABC):
    """Base retrieval object"""
    @abstractmethod
    def run_retrieval(self,
                      retrieval_inputs: RetrievalInput):
        pass

    @staticmethod
    def generate_theta_0(ig):
        
        if hasattr(ig, "__len__"):
            if hasattr(ig[0], "__len__"):
                theta_0 = np.concatenate(ig).flatten()
            else:
                theta_0 = np.array(ig).flatten()
        else:
            theta_0 = np.array([ig])
            
        return theta_0
    
if __name__ == "__main__":
    pass