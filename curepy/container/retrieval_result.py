"""Container class for retrieval results"""
import xarray as xr
import obsarray
from typing import Optional
import numpy as np

class RetrievalResult:
    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        u_x: Optional[np.ndarray] = None,
        corr_x: Optional[np.ndarray] = None,
        samples: Optional[np.ndarray] = None,
        b_samples: Optional[np.ndarray] = None,
        x_names: Optional[list] = None,
    ) -> None:
        """
        Container for retrieval output quantities.

        :param x: Retrieved state vector.
        :param u_x: Uncertainty of the retrieved state vector.
        :param corr_x: Error-correlation matrix of the retrieved state vector.
        :param samples: MCMC samples of the retrieved state vector.
        :param b_samples: MCMC samples of the ancillary parameters.
        :param x_names: Names of the retrieved quantities.
        """

        self.values = x
        self.uncertainties = u_x
        self.correlation = corr_x
        self.samples = samples
        self.b_samples = b_samples
        self.x_names = x_names

    def build_obsarray(self) -> None:
        """
        Build an ``obsarray`` dataset from the retrieval results.

        .. note::
            Not yet implemented.
        """
        pass