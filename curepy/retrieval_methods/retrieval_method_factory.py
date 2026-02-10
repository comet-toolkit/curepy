"""Factory design to generate retrieval objects"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.retrieval_methods.MCMC import MCMC
from curepy.retrieval_methods.LPU import LPU

from typing import Union

RETRIEVAL_HANDLERS = {
    "mcmc": MCMC,
    "lpu": LPU
    
} 

class RetrievalFactory:
    def __init__(
        self
    ):
        self.retrieval_objects = RETRIEVAL_HANDLERS
    
    def make_retrieval_object(self,
                              name: Union[str, BaseRetrieval],
                              *args, **kwargs) -> BaseRetrieval:
        """Return specified retrieval object

        :param name: Selected retrieval method (eg. "mcmc")
        :return: Retrieval method object
        """
        
        if name in self.retrieval_objects.values():
            return name(*args, **kwargs)
        
        elif name.lower() in self.retrieval_objects.keys():
            return self.retrieval_objects[name.lower()](*args, **kwargs)
        
        else:
            raise ValueError(
                f"The provided retrieval name ({name}) is not an implemented retrieval method {self.retrieval_objects.keys()}"
            )
        