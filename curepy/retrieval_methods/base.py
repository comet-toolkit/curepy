"""Base class for retrieval methods"""

from abc import ABC, abstractmethod
from curepy.container.retrieval_input import RetrievalInput

class BaseRetrieval(ABC):
    """Base retrieval object"""
    
    def __init__(self, 
                 parallel_cores = 1,
    ):
        
        self.parallel_cores = parallel_cores
    
    @abstractmethod
    def run_retrieval(self,
                      retrieval_inputs: RetrievalInput):
        pass

if __name__ == "__main__":
    pass