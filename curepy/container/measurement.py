"""Container for Measurement data"""

import numpy as np
import comet_maths as cm
import curepy.utilities.utilities as util


class Measurement:
    def __init__(
        self,
        y,
        u_y=None,
        corr_y=None,
    ):
        """
        Container class for Measurement variable data

        :param y: Measurement variable
        :param u_y: Uncertainty of measurement variable, must be the same shape as y
        :param corr_y: Error-correlation of measurement variable, str("rand" or "syst") or square matrix with side length equal to length of measurement variable input
        """

        self.y = y
        self.u_y = u_y
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
    def _flatten_inputs(y, u_y):
        y_flat, y_shape = util.flatten_array(y)
        if u_y is not None:
            u_y_flat, u_y_shape = util.flatten_array(u_y)
        if not y_shape == u_y_shape:
            raise ValueError(
                "Measurement variable, y, and related uncertainties, u_y, have different shapes:",
                y_shape,
                u_y_shape,
            )

        return y_flat, u_y_flat, y_shape

    @staticmethod
    def _check_shapes(y, u_y, corr_y):
        """
        Check shapes of measurement variable y, uncertainties, and error correlations are compatible.
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
    def calculate_inv_cov(unc, corr):
        """
        Calculate inverse covariance matrix

        :param unc: Uncertainty
        :param corr: Correlation matrix
        """

        cov = cm.convert_corr_to_cov(corr, unc)

        if np.array_equal(cov, np.diag(np.diag(cov))):
            return np.diag(1 / np.diag(cov))
        else:
            # might need a check for PD here
            return np.linalg.inv(cov)
