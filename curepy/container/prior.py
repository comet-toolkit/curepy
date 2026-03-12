"""Container for Prior information"""

import numpy as np
from curepy.utilities.distributions import *
from typing import List, Optional
import warnings
import curepy.utilities.utilities as util
import comet_maths as cm

implemented_prior_shapes = {
    "uniform": {
        "function": ln_uniform,
        "params": ["minimum", "maximum"],
        "correlation": False,
    },
    "normal": {"function": ln_normal, "params": ["mu", "sigma"], "correlation": True},
}


class Prior:
    def __init__(
        self,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Container for prior probability distribution information.

        :param prior_shape: List of prior distribution shape names, one per
            retrieval parameter.  Supported values: ``"uniform"``,
            ``"normal"``.
        :param prior_params: List of parameter dictionaries, one per
            retrieval parameter.  Required keys depend on ``prior_shape``:
            ``"uniform"`` requires ``"minimum"`` and ``"maximum"``;
            ``"normal"`` requires ``"mu"`` and ``"sigma"``.
        :param prior_correlation: Correlation matrix between retrieval
            parameters, or ``None`` to assume no correlation.
        :raises ValueError: If ``prior_shape`` is ``None``, contains
            unsupported distribution names, required parameters are missing,
            or the correlation is incompatible with the chosen distribution.
        """

        self._check_inputs(prior_shape, prior_params, prior_correlation)
        self.function_list = [
            implemented_prior_shapes[shape]["function"] for shape in prior_shape
        ]
        self.prior_params = prior_params
        self.prior_correlation = prior_correlation
        self.Sa_inv = self.return_Sa_inv()

        if np.all(prior_shape == "normal"):
            mu = np.array([p["mu"] for p in prior_params])
            self.lnprior = lambda x: ln_multi_normal(x, mu, self.Sa_inv)
        else:
            self.lnprior = self.combine_dist_functions

    @staticmethod
    def _check_inputs(
        shape: List[str],
        params: List[dict],
        corr: Optional[np.ndarray],
    ) -> None:
        """
        Validate the prior shape, parameter, and correlation inputs.

        :param shape: List of prior distribution shape names.
        :param params: List of parameter dictionaries for each distribution.
        :param corr: Correlation matrix between retrieval parameters, or
            ``None``.
        :raises ValueError: If any shape is unsupported, required parameters
            are missing, or a correlation is provided for a distribution that
            does not support it.
        """
        if shape is None:
            raise ValueError(
                "Cannot instantiate empty Prior object"
            )
        # check shape
        if any([shape.lower() not in implemented_prior_shapes for shape in shape]):
            raise ValueError(
                f"A provided prior shape ({shape}) is not an implemented prior distribution shape {implemented_prior_shapes.keys()}"
            )
        # check required params exist
        for i, sh in enumerate(shape):
            if sorted(list((params[i].keys()))) != sorted(
                implemented_prior_shapes[sh]["params"]
            ):
                raise ValueError(
                    f"Prior shape ({sh}) requires the following inputs in param dictionary {implemented_prior_shapes[sh]['params']}"
                )

            if corr is not None and not implemented_prior_shapes[sh]["correlation"]:
                raise ValueError(
                    f"Prior shape ({sh}) is not compatible with correlation inputs"
                )
            if corr is None and implemented_prior_shapes[sh]["correlation"]:
                warnings.warn(
                    "No correlation between priors set, assuming random correlation"
                )

    def combine_dist_functions(self, xs: np.ndarray):
        """
        Evaluate each individual prior distribution at the corresponding value.

        :param xs: Sequence of parameter values, one per retrieval parameter.
        :returns: Callable that, when invoked, returns a list of log-prior
            values from each individual distribution.
        """
        return lambda: [
            f(x, **kws) for f, x, kws in zip(self.function_list, xs, self.prior_params)
        ]

    def return_Sa_inv(self) -> Optional[np.ndarray]:
        """
        Compute the inverse of the prior covariance matrix.

        Constructs the prior covariance from the ``"sigma"`` entries in
        ``prior_params`` and the (optionally formatted) correlation matrix,
        then returns its inverse.  Returns ``None`` if no correlation
        matrix is available.

        :returns: Inverse of the prior covariance matrix, or ``None`` if
            the correlation matrix is undefined.
        """
        corr = util.format_correlation(self.prior_params, self.prior_correlation)
        if corr is None:
            return None
        else:
            u = np.array(
                [p["sigma"] for p in self.prior_params]
            )  # only set up for gaussian priors
            Sa = cm.convert_corr_to_cov(corr, u)
            return np.linalg.inv(Sa)
