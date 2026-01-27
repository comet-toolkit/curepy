"""Base class for retrieval methods"""

from abc import ABC, abstractmethod

class BaseRetrieval(ABC):
    """Base retrieval object"""
    
    def __init__(
        self,
    ):
        raise NotImplementedError