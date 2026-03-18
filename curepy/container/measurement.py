"""Container for Measurement data"""

import numpy as np
from typing import Optional, Union
import comet_maths as cm
import curepy.utilities.utilities as util


class Measurement:
    def __init__(
        self,
        y: np.ndarray,
        u_y_total: Optional[np.ndarray] = None,
        u_y_rand: Optional[np.ndarray] = None,
        u_y_syst: Optional[np.ndarray] = None,
        corr_y: Optional[Union[str, np.ndarray]] = None,
    ) -> None:
        """
        Container class for measurement variable data.

        :param y: Measurement variable.
        :param u_y_total: Total uncertainty of measurement variable; must have the same
            shape as ``y``.
        :param u_y_rand: Random uncertainty of measurement variable; must have the same
            shape as ``y``.
        :param u_y_syst: Systematic uncertainty of measurement variable; must have the same
            shape as ``y``.
        :param corr_y: Error-correlation of the measurement variable.
            Accepted values: ``None``, ``"rand"`` (random), ``"syst"``
            (systematic), or a square matrix whose side length equals the
            length of ``y``.
        """

        u_y_total, corr_y = self._format_uncertainty(
            u_y_total, u_y_rand, u_y_syst, corr_y
        )

        self.y = y
        self.u_y = u_y_total
        self.y_flat, self.u_y_flat, self.y_shape = self._flatten_inputs(
            self.y, self.u_y
        )

        self.corr_y = util.format_correlation(self.y_flat, corr_y)

        self._check_shapes(self.y_flat, self.u_y_flat, self.corr_y)

        if corr_y is not None:
            self.invcov = self.calculate_inv_cov(self.u_y_flat, self.corr_y)
        else:
            self.invcov = None

    @staticmethod
    def _flatten_inputs(
        y: np.ndarray,
        u_y: Optional[np.ndarray],
    ) -> tuple:
        """
        Flatten the measurement variable and its uncertainties.

        :param y: Measurement variable array.
        :param u_y: Uncertainty array for the measurement variable, or
            ``None`` if uncertainties are not provided.
        :returns: Tuple of ``(y_flat, u_y_flat, y_shape)`` where
            ``u_y_flat`` is ``None`` when ``u_y`` is ``None``.
        """
        y_flat, y_shape = util.flatten_array(y)
        if u_y is not None:
            u_y_flat, u_y_shape = util.flatten_array(u_y)
            if y_shape != u_y_shape:
                raise ValueError(
                    "Measurement variable, y, and related uncertainties, u_y, have different shapes:",
                    y_shape,
                    u_y_shape,
                )
        else:
            u_y_flat = None

        return y_flat, u_y_flat, y_shape

    @staticmethod
    def _check_shapes(
        y: np.ndarray,
        u_y: Optional[np.ndarray],
        corr_y: Optional[np.ndarray],
    ) -> None:
        """
        Check that the shapes of ``y``, ``u_y``, and ``corr_y`` are
        mutually compatible.

        :param y: Flattened measurement variable.
        :param u_y: Flattened uncertainty array, or ``None``.
        :param corr_y: Error-correlation matrix, or ``None``.
        """
        N = len(y)

        if u_y is not None and len(u_y) != N:
            raise ValueError(
                "Length of measured variable y must match length of uncertainty variable"
            )

        if u_y is None and corr_y is not None:
            raise ValueError(
                "Uncertainties must be defined if error correlation matrix is defined"
            )

        if corr_y is not None and corr_y.shape[0] != corr_y.shape[1]:
            raise ValueError("Error correlation matrix must be a square matrix")

        if corr_y is not None and corr_y.shape[0] != N:
            raise ValueError(
                "Length of measured variable y must match side length of error correlation matrix"
            )

    @staticmethod
    def _format_uncertainty(u_total, u_rand, u_syst, corr):

        if u_total is None and u_rand is None and u_syst is None:
            return None, None
        elif u_total is not None and u_rand is None and u_syst is None:
            return u_total, corr
        elif u_total is not None and (u_rand is not None or u_syst is not None):
            raise ValueError("If u_y_total is set, u_y_rand and u_y_syst must be None")
        elif u_total is None and u_rand is not None and u_syst is None:
            return u_rand, "rand"
        elif u_total is None and u_rand is None and u_syst is not None:
            return u_syst, "syst"
        elif u_total is None and u_rand is not None and u_syst is not None:
            tot = np.sqrt(u_rand**2 + u_syst**2)
            tot_cov = cm.convert_corr_to_cov(
                np.eye(len(u_rand.flatten())), u_rand.flatten()
            ) + cm.convert_corr_to_cov(
                np.ones((len(u_syst.flatten()), len(u_syst.flatten()))),
                u_syst.flatten(),
            )
            tot_corr = cm.convert_cov_to_corr(tot_cov, tot)
            return tot, tot_corr

    @staticmethod
    def calculate_inv_cov(unc: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """
        Calculate the inverse covariance matrix.

        :param unc: Uncertainty (standard deviation) array.
        :param corr: Correlation matrix.
        :returns: Inverse of the covariance matrix.
        """

        cov = cm.convert_corr_to_cov(corr, unc)

        if np.array_equal(cov, np.diag(np.diag(cov))):
            return np.diag(1 / np.diag(cov))
        else:
            # might need a check for PD here
            return np.linalg.inv(cov)
