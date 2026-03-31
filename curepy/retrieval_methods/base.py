"""Base class for retrieval methods"""

from abc import ABC, abstractmethod
from curepy.container.retrieval_input import RetrievalInput
import numpy as np
from curepy.utilities.maths import lnlike
import warnings
from typing import Optional, List


class BaseRetrieval(ABC):
    """Base retrieval object."""

    @abstractmethod
    def _run_retrieval(self, retrieval_inputs: RetrievalInput):
        """
        Abstract run_retrieval method to be implemented by each retrieval subclass

        :param retrieval_inputs: Object encapsulating all inputs needed for
            the retrieval.
        """
        pass

    def run_retrieval(self, retrieval_inputs: RetrievalInput, *args, **kwargs):
        """
        Execute the retrieval algorithm.

        :param retrieval_inputs: Object encapsulating all inputs needed for
            the retrieval.
        """
        result = self._run_retrieval(retrieval_inputs, *args, **kwargs)

        return result

    def reshape_outputs(
        self,
        x: np.ndarray,
        u_x: np.ndarray,
        corr_x: Optional[np.ndarray],
    ) -> tuple:
        """
        Reshape flat retrieval outputs back to the initial-guess shape.

        :param x: Flat retrieved state vector.
        :param u_x: Flat uncertainties of the retrieved state vector.
        :param corr_x: Correlation matrix of the retrieved state vector,
            or ``None``.  Reshaping of correlation matrices is not yet
            implemented; a warning is emitted when ``corr_x`` is not ``None``.
        :returns: Tuple of ``(x, u_x, corr_x)`` reshaped to the
            initial-guess shape.
        """
        x_shape = self.retrieval_input.measurement_function_obj.initial_guess.shape

        x = x.reshape(x_shape)
        u_x = u_x.reshape(x_shape)
        if corr_x is not None:
            warnings.warn("Reshaping of correlation matrices is not yet implemented")
        # corr_x = corr_x.reshape(x.shape + (np.prod(x_shape),))
        return x, u_x, corr_x

    @staticmethod
    def generate_theta_0(ig: np.ndarray) -> np.ndarray:
        """
        Convert the initial guess into a flat 1-D state vector.

        :param ig: Initial guess, which may be a scalar, 1-D array, or
            2-D array (one row per measurement location).
        :returns: Flat 1-D initial state vector.
        """

        if hasattr(ig, "__len__"):
            if hasattr(ig[0], "__len__"):
                try:
                    theta_0 = np.concatenate(ig).flatten()
                except:
                    theta_0 = np.concatenate([x.flatten() for x in ig])
            else:
                theta_0 = np.array(ig).flatten()
        else:
            theta_0 = np.array([ig])

        return theta_0

    def _check_retrieval_input(self) -> None:
        """
        Validate and fill in default retrieval sub-objects.

        Builds a default (no ancillary parameters) ancillary object if one
        has not been provided, and builds a flat uniform prior over all
        retrieval parameters if no prior has been set.
        """

        # format and define retrieval input
        if self.retrieval_input.ancillary_obj is None:
            self.retrieval_input.build_ancillary()
        if self.retrieval_input.prior_obj is None:
            self.retrieval_input.build_prior(
                prior_shape=["uniform"]
                * len(self.retrieval_input.measurement_function_obj.initial_guess),
                prior_params=[{"minimum": -np.inf, "maximum": np.inf}]
                * len(self.retrieval_input.measurement_function_obj.initial_guess),
            )

    def find_chisum(
        self,
        theta: np.ndarray,
        repeat_dims: List[int] = [],
    ) -> float:
        """
        Compute the chi-squared cost between the forward model and observations.

        Evaluates the measurement function at ``theta``, computes the
        residual with respect to the observations, and returns the weighted
        sum of squared residuals using the inverse covariance matrix (or
        diagonal uncertainties when no covariance is available).

        :param theta: Current retrieval state vector.
        :param repeat_dims: Indices of repeat dimensions along which to
            accumulate the chi-squared sum.  Only zero or one repeat
            dimensions are currently supported.
        :returns: Chi-squared cost value.
        """

        modelled_data = (
            self.retrieval_input.measurement_function_obj.measurement_function_x(
                theta, self.retrieval_input.ancillary_obj.b
            ).flatten()
        )
        diff = modelled_data - self.retrieval_input.measurement_obj.y_flat
        if np.isfinite(np.sum(diff)):
            if self.retrieval_input.measurement_obj.invcov is None:
                return np.sum((diff) ** 2 / self.retrieval_input.measurement_obj.u_y_flat**2)
            else:
                if len(repeat_dims) == 0:
                    return np.dot(
                        np.dot(diff.T, self.retrieval_input.measurement_obj.invcov),
                        diff,
                    )
                elif len(repeat_dims) == 1:
                    sum = 0
                    for i in range(diff.shape[repeat_dims[0]]):
                        diffi = np.take(diff, i, repeat_dims[0])
                        sum += np.dot(
                            np.dot(
                                diffi.T, self.retrieval_input.measurement_obj.invcov
                            ),
                            diffi,
                        )
                    return sum
                else:
                    raise ValueError(
                        "Methods for multiple repeat dimensions are not yet implemented,"
                    )
        else:
            print("The difference between model and observations is infinite")
        return np.inf

    def lnprob(self, theta: np.ndarray) -> float:
        """
        Compute the log posterior probability for state vector ``theta``.

        Evaluates the log prior and the log likelihood and returns their sum.
        Returns ``-np.inf`` if the prior is not finite at ``theta``.

        :param theta: Current retrieval state vector.
        :returns: Log posterior probability.
        """
        lp_prior = self.retrieval_input.prior_obj.lnprior(
            theta,
        )()
        if not all(np.isfinite(lp_prior)):
            return -np.inf

        lp = lnlike(self.find_chisum(theta, repeat_dims=[]))

        return np.sum(lp_prior) + lp


if __name__ == "__main__":
    pass
