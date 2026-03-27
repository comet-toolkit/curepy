"""Optimal Estimation retrieval class"""

from curepy.retrieval_methods.base import BaseRetrieval
from curepy.container.retrieval_input import RetrievalInput
from curepy.container.retrieval_result import RetrievalResult

from scipy.optimize import minimize
import numpy as np
import comet_maths as cm
from functools import partial
from typing import Optional, Callable


class OE(BaseRetrieval):
    """Optimal Estimation (OE) retrieval object."""

    def __init__(self, Jx: Optional[np.ndarray] = None) -> None:
        """
        Initialise the OE retrieval object.

        :param Jx: Pre-computed Jacobian of the measurement function with
            respect to the state vector.  If ``None``, the Jacobian is
            computed numerically during :meth:`run_retrieval`.
        """

        self.Jx = Jx

    def _run_retrieval(
        self,
        retrieval_input: RetrievalInput,
        return_corr: bool = True,
        reshape_results: bool = False,
    ) -> RetrievalResult:
        """
        Run the OE retrieval.

        Minimises the negative log posterior with
        :func:`scipy.optimize.minimize`, then propagates measurement and
        ancillary-parameter uncertainties through the inverse Jacobian to
        obtain state-vector uncertainties.

        :param retrieval_input: Object containing all retrieval inputs.
        :param return_corr: If ``True``, include the state-vector correlation
            matrix in the result.
        :param reshape_results: If ``True``, reshape the flat output arrays
            to the initial-guess shape.
        :returns: Retrieved values, uncertainties, and optionally the
            correlation matrix.
        """

        self.retrieval_input = retrieval_input

        self._check_retrieval_input()

        theta_0 = self.generate_theta_0(
            self.retrieval_input.measurement_function_obj.initial_guess
        )

        res = minimize(-self.lnprob, theta_0)

        if self.Jx is None:
            Jx = self.calculate_Jx(res.x)
        else:
            Jx = self.Jx

        x = res.x
        u_func, corr_x = self.process_inverse_jacobian(Jx, res.x)

        if reshape_results:
            x, u_func, corr_x = self.reshape_outputs(x, u_func, corr_x)

        return RetrievalResult(
            x=x,
            u_x=u_func,
            corr_x=corr_x if return_corr else None,
            x_names=self.retrieval_input.measurement_function_obj._input_quantities_names,
        )

    def process_inverse_jacobian(
        self,
        J: np.ndarray,
        x: np.ndarray,
    ) -> tuple:
        """
        Derive state-vector uncertainties from the Jacobian via LPU.

        :param J: Jacobian of the measurement function with respect to the
            state vector, evaluated at ``x``.
        :param x: Retrieved state vector.
        :returns: Tuple of ``(u_func, corr_x)`` where ``u_func`` is the
            1-sigma uncertainty array and ``corr_x`` is the correlation matrix.
        """
        covx = self.calculate_measurand_covariance(
            x,
            J,
            self.retrieval_input.measurement_obj.invcov,
            Sa_inv=self.retrieval_input.prior_obj.Sa_inv,
        )
        u_func = np.sqrt(np.diag(covx))
        corr_x = cm.convert_cov_to_corr(covx, u_func)

        return u_func, corr_x

    def calculate_measurand_covariance(
        self,
        x: np.ndarray,
        J: np.ndarray,
        Sy_inv: Optional[np.ndarray],
        Sa_inv: Optional[np.ndarray] = None,
        Sb_inv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate the posterior state-vector covariance matrix.

        Uses the Gauss–Newton / LPU formula combining measurement,
        ancillary, and prior uncertainty contributions.

        :param x: Retrieved state vector.
        :param J: Jacobian with respect to the state vector.
        :param Sy_inv: Inverse measurement covariance.  Must not be ``None``
            unless ``Sb_inv`` is also provided.
        :param Sa_inv: Inverse prior covariance, or ``None`` if no prior is
            used.
        :param Sb_inv: Pre-computed inverse ancillary-parameter covariance
            mapped to measurement space.  If ``None``, the covariance is
            computed from the ancillary object.
        :returns: Posterior state-vector covariance matrix.
        """

        if Sy_inv is not None and Sb_inv is not None:
            Se_inv = Sy_inv + Sb_inv

        elif Sy_inv is not None and Sb_inv is None:
            Sb = self.retrieval_input.ancillary_obj.calculate_b_cov()
            if Sb is not None:
                Jb = self.calculate_Jb(x)
                Sb_y = np.dot(np.dot(Jb, Sb), Jb.T)
                Se = np.linalg.inv(Sy_inv) + Sb_y
            else:
                Se = np.linalg.inv(Sy_inv)

            Se_inv = np.linalg.inv(Se)

        else:
            raise ValueError("Covariance must be set for LPU method")

        if Sa_inv is not None:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J) + Sa_inv)
        else:
            return np.linalg.inv(np.dot(np.dot(J.T, Se_inv), J))

    def calculate_Jx(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically compute the Jacobian of the measurement function with
        respect to the state vector.

        :param x: State vector at which the Jacobian is evaluated.
        :returns: Jacobian matrix.
        """

        meas_func_fixed_b = partial(
            self.retrieval_input.measurement_function_obj.measurement_function_flattened_output,
            b=self.retrieval_input.ancillary_obj.b,
        )

        Jx = cm.calculate_Jacobian(meas_func_fixed_b, x)

        return Jx

    def calculate_Jb(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically compute the Jacobian of the measurement function with
        respect to the flattened ancillary parameters.

        :param x: State vector at which the Jacobian is evaluated.
        :returns: Jacobian matrix.
        """

        b_flat = np.hstack([b.flatten() for b in self.retrieval_input.ancillary_obj.b])
        b_shape_list = [b.shape for b in self.retrieval_input.ancillary_obj.b]

        meas_func_fixed_x = lambda b: self.retrieval_input.measurement_function_obj.measurement_function_flattened_b(
            x, b, b_shape_list
        )

        Jb = cm.calculate_Jacobian(meas_func_fixed_x, b_flat)

        return Jb
