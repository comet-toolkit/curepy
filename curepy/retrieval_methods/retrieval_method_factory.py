"""Factory design to generate retrieval objects"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.retrieval_methods.mcmc import MCMC
from curepy.retrieval_methods.optimal_estimation import OE

from typing import Union

RETRIEVAL_HANDLERS = {"mcmc": MCMC, "oe": OE}


class RetrievalFactory:
    def __init__(self) -> None:
        """
        Initialise the factory with the default set of retrieval handlers.
        """
        self.retrieval_objects = RETRIEVAL_HANDLERS

    def make_retrieval_object(
        self, name: Union[str, BaseRetrieval], *args, **kwargs
    ) -> BaseRetrieval:
        """
        Return the specified retrieval object.

        :param name: Retrieval method identifier.  May be a string key
            (e.g. ``"mcmc"``, ``"oe"``) or a
            :class:`~curepy.retrieval_methods.base.BaseRetrieval` subclass.
        :returns: Instantiated retrieval method object.
        """

        if name in self.retrieval_objects.values():
            return name(*args, **kwargs)

        elif name.lower() in self.retrieval_objects.keys():
            return self.retrieval_objects[name.lower()](*args, **kwargs)

        else:
            raise ValueError(
                f"The provided retrieval name ({name}) is not an implemented retrieval method {self.retrieval_objects.keys()}"
            )
