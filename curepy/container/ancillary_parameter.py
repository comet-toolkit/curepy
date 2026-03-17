"""Container for ancillary parameter information"""

import numpy as np
from typing import Optional
import punpy
import comet_maths as cm
import warnings
import curepy.utilities.utilities as util
from scipy.linalg import block_diag


class AncillaryParameter:
    def __init__(
        self,
        b: Optional[list] = None,
        u_b: Optional[list] = None,
        corr_b: Optional[list] = None,
        corr_between_b: Optional[np.ndarray] = None,
        b_samples: Optional[np.ndarray] = None,
        b_MC_steps: Optional[int] = None,
    ) -> None:
        """
        Container for ancillary (forward-model) parameter data and uncertainties.

        :param b: List of ancillary parameter arrays.
        :param u_b: List of uncertainty arrays for each ancillary parameter.
        :param corr_b: List of error-correlation specifications for each
            ancillary parameter.  Each element may be ``None``, ``"rand"``,
            ``"syst"``, or a square correlation matrix.
        :param corr_between_b: Correlation matrix between the different
            ancillary parameters.
        :param b_samples: Pre-computed Monte Carlo samples for the ancillary
            parameters.
        :param b_MC_steps: Number of Monte Carlo steps to use when generating
            ancillary parameter samples.
        """

        self.b = None
        self.u_b = None
        self.corr_b = None
        self.corr_between_b = None

        if b is not None:
            self._format_ancillary_data(b, u_b, corr_b, corr_between_b)

        self.b_MC_steps = b_MC_steps
        self.b_samples = b_samples
        

    def _format_ancillary_data(
        self,
        b: list,
        u_b: Optional[list],
        corr_b: Optional[list],
        corr_between_b: Optional[np.ndarray],
    ) -> None:
        """
        Validate and format ancillary parameter inputs into numpy arrays.

        Converts scalar values to 1-D arrays, handles ragged
        (non-rectangular) parameter lists, and formats correlation matrices
        via :func:`~curepy.utilities.utilities.format_correlation`.

        :param b: List of ancillary parameter arrays.
        :param u_b: List of uncertainty arrays for each ancillary parameter,
            or ``None``.
        :param corr_b: List of error-correlation specifications, or ``None``.
        :param corr_between_b: Correlation matrix between ancillary
            parameters, or ``None``.
        """
        multiple_b = False
        if b is not None:
            for i in range(len(b)):
                if not hasattr(b[i], "__len__"):
                    b[i] = np.array([b[i]])
                else:
                    b[i] = np.array(b[i])
            try:
                self.b = np.array(b)
            except:
                multiple_b = True
                self.b = util.to_ragged_array(b)

        if u_b is not None:
            for i in range(len(u_b)):
                if u_b[i] is None:
                    u_b[i] = np.zeros_like(self.b[i])
                elif not hasattr(u_b[i], "__len__"):
                    u_b[i] = np.array([u_b[i]])
                else:
                    u_b[i] = np.array(u_b[i])

            if multiple_b:
                self.u_b = util.to_ragged_array(u_b)
            else:
                self.u_b = u_b

        if corr_b is not None:
            if multiple_b:
                self.corr_b = util.to_ragged_array(
                    [
                        util.format_correlation(self.b[i], corr_b[i])
                        for i in range(self.b.shape[0])
                    ]
                )
            else:
                self.corr_b = util.format_correlation(self.b, corr_b)

        if corr_between_b is not None:
            self.corr_between_b = np.array(corr_between_b)

    def generate_b_samples(self):
        """
        Generate Monte Carlo samples for the ancillary parameters.

        Uses :class:`punpy.MCPropagation` to draw samples from the joint
        distribution defined by ``b``, ``u_b``, ``corr_b``, and
        ``corr_between_b``.  If ``b`` is ``None`` the resulting
        ``b_samples`` attribute is set to ``None``.  If ``b_samples`` has
        already been provided it is left unchanged.
        """
        if self.b is None:
            self.b_samples = None
        else:
            if self.b_samples is not None:
                self.b_samples = self.b_samples
            else:
                if self.u_b is not None:
                    prop = punpy.MCPropagation(self.b_MC_steps)
                    if self.b_samples is None:
                        self.b_samples = prop.generate_MC_sample(
                            self.b,
                            self.u_b,
                            self.corr_b,
                            self.corr_between_b,
                        )
                else:
                    self.b_samples = self.b

    def calculate_b_cov(self) -> Optional[np.ndarray]:
        """
        Calculate the full covariance matrix for all ancillary parameters.

        Constructs per-parameter correlation matrices, flattens and combines
        them with ``corr_between_b`` (if set) using
        ``comet_maths.calculate_flattened_corr``.  Returns ``None`` if
        insufficient data is available and emits a warning.

        :returns: Combined covariance matrix for all ancillary parameters,
            or ``None`` if it cannot be computed.
        """
        if (
            self.b is None
            and self.u_b is None
            and (self.corr_b is None or self.corr_between_b is None)
        ):
            warnings.warn(
                "b, u_b, and a correlation matrix must be defined to calculate a covariance matrix"
            )
            return None
        else:
            if self.corr_b is not None:
                for i in range(len(self.corr_b)):
                    if self.corr_b[i] is None:
                        self.corr_b[i] = np.eye(len(self.b[i]))
                if len(set([len(corr) for corr in self.corr_b])) == 1:
                    total_corr = cm.calculate_flattened_corr(
                        corrs=[corr for corr in self.corr_b],
                        corr_between=(
                            self.corr_between_b
                            if self.corr_between_b is not None
                            else np.eye(len(self.b))
                        ),
                    )
                else:
                    total_corr = block_diag(*[corr for corr in self.corr_b])
                    warnings.warn(
                        "Correlation matrices for each b have different shapes. Assuming no correlation between different b values."
                    )

                return cm.convert_corr_to_cov(
                    total_corr, np.hstack([b.flatten() for b in self.u_b])
                )
            else:
                warnings.warn(
                    "Correlation matrix must be defined to calculate covariance"
                )
                return None
