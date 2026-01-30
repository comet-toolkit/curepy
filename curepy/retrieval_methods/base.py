"""Base class for retrieval methods"""

from abc import ABC, abstractmethod
from curepy.container.retrieval_input import RetrievalInput

class BaseRetrieval(ABC):
    """Base retrieval object"""
    @abstractmethod
    def run_retrieval(self,
                      retrieval_inputs: RetrievalInput):
        pass

if __name__ == "__main__":
    pass