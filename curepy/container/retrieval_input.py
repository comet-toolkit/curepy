"""Container for all retrieval inputs"""

from curepy.container.measurement_function import MeasurementFunction
from curepy.container.ancillary_parameter import AncillaryParameter
from curepy.container.prior import Prior
from curepy.container.measurement import Measurement

from typing import Union, List, Optional, Callable, Any


class RetrievalInput:

    def __init__(
        self,
        measurement_function_obj: Optional[MeasurementFunction] = None,
        measurement_obj: Optional[Measurement] = None,
        ancillary_obj: Optional[AncillaryParameter] = None,
        prior_obj: Optional[Prior] = None,
    ) -> None:
        """
        Container aggregating all inputs required for a retrieval.

        Individual sub-objects can be supplied directly or built afterwards
        using the ``build_*`` helper methods.

        :param measurement_function_obj: Measurement function container.
        :param measurement_obj: Measurement data container.
        :param ancillary_obj: Ancillary parameter container.
        :param prior_obj: Prior distribution container.
        """

        self.measurement_function_obj = measurement_function_obj
        self.measurement_obj = measurement_obj
        self.ancillary_obj = ancillary_obj
        self.prior_obj = prior_obj

    def build_retrieval_inputs(
        self,
        measurement_func: Callable,
        initial_guess: Any,
        y: Any,
        u_y_total: Optional[Any] = None,
        u_y_rand: Optional[Any] = None,
        u_y_syst: Optional[Any] = None,
        corr_y: Optional[Union[str, Any]] = None,
        multiple_guess_measurements: bool = False,
        measurement_name: str = None,
        input_quantities_names: Union[str, List[str]] = None,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation: Optional[Any] = None,
        b: Optional[list] = None,
        u_b: Optional[list] = None,
        corr_b: Optional[list] = None,
        corr_between_b: Optional[Any] = None,
        b_samples: Optional[Any] = None,
        b_MC_steps: Optional[int] = None,
    ) -> None:
        """
        Construct all retrieval input sub-objects in a single call.

        :param measurement_func: Callable measurement/forward-model function.
        :param initial_guess: Initial values for the retrieval parameters.
        :param y: Measurement variable.
        :param u_y_total: Total uncertainty of the measurement variable.
        :param u_y_rand: Random uncertainty of the measurement variable.
        :param u_y_syst: Systematic uncertainty of the measurement variable.
        :param corr_y: Error-correlation of the measurement variable
            (``None``, ``"rand"``, ``"syst"``, or a square matrix).
        :param multiple_guess_measurements: If ``True``, the initial guess
            contains multiple measurements per parameter.
        :param measurement_name: Optional name for the measured quantity.
        :param input_quantities_names: Optional name(s) for input quantities.
        :param prior_shape: List of prior distribution shape names.
        :param prior_params: List of prior parameter dictionaries.
        :param prior_correlation: Correlation matrix for the prior.
        :param b: Ancillary parameter values.
        :param u_b: Uncertainties for the ancillary parameters.
        :param corr_b: Correlation specification for each ancillary parameter.
        :param corr_between_b: Correlation matrix between ancillary parameters.
        :param b_samples: Pre-generated MC samples for ancillary parameters.
        :param b_MC_steps: Number of MC steps for ancillary parameter sampling.
        """

        self.measurement_function_obj = MeasurementFunction(
            measurement_func,
            initial_guess,
            multiple_guess_measurements,
            measurement_name,
            input_quantities_names,
        )

        self.measurement_obj = Measurement(y, u_y_total, u_y_rand, u_y_syst, corr_y)

        self.ancillary_obj = AncillaryParameter(
            b, u_b, corr_b, corr_between_b, b_samples, b_MC_steps
        )
        self.prior_obj = Prior(prior_shape, prior_params, prior_correlation)

    def build_measurement_function(
        self,
        measurement_func: Callable,
        initial_guess: Any,
        multiple_guess_measurements: bool = False,
        measurement_name: str = None,
        input_quantities_names: Union[str, List[str]] = None,
    ) -> None:
        """
        Construct ``measurement_function_obj`` from individual components.

        :param measurement_func: Callable measurement/forward-model function.
        :param initial_guess: Initial values for the retrieval parameters.
        :param multiple_guess_measurements: If ``True``, the initial guess
            contains multiple measurements per parameter.
        :param measurement_name: Optional name for the measured quantity.
        :param input_quantities_names: Optional name(s) for input quantities.
        """

        self.measurement_function_obj = MeasurementFunction(
            measurement_func,
            initial_guess,
            multiple_guess_measurements,
            measurement_name,
            input_quantities_names,
        )

    def build_measurement(
        self,
        y: Any,
        u_y_total: Optional[Any] = None,
        u_y_rand: Optional[Any] = None,
        u_y_syst: Optional[Any] = None,
        corr_y: Optional[Union[str, Any]] = None,
    ) -> None:
        """
        Construct ``measurement_obj`` from measurement data.

        :param y: Measurement variable.
        :param u_y_total: Total uncertainty of the measurement variable.
        :param u_y_rand: Random uncertainty of the measurement variable.
        :param u_y_syst: Systematic uncertainty of the measurement variable.
        :param corr_y: Error-correlation of the measurement variable
            (``None``, ``"rand"``, ``"syst"``, or a square matrix).
        """

        self.measurement_obj = Measurement(y, u_y_total, u_y_rand, u_y_syst, corr_y)

    def build_prior(
        self,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation: Optional[Any] = None,
    ) -> None:
        """
        Construct ``prior_obj`` from prior distribution specifications.

        :param prior_shape: List of prior distribution shape names.
        :param prior_params: List of prior parameter dictionaries.
        :param prior_correlation: Correlation matrix for the prior.
        """

        self.prior_obj = Prior(prior_shape, prior_params, prior_correlation)

    def build_ancillary(
        self,
        b: Optional[list] = None,
        u_b: Optional[list] = None,
        corr_b: Optional[list] = None,
        corr_between_b: Optional[Any] = None,
        b_samples: Optional[Any] = None,
        b_MC_steps: Optional[int] = None,
    ) -> None:
        """
        Construct ``ancillary_obj`` from ancillary parameter data.

        :param b: Ancillary parameter values.
        :param u_b: Uncertainties for the ancillary parameters.
        :param corr_b: Correlation specification for each ancillary parameter.
        :param corr_between_b: Correlation matrix between ancillary parameters.
        :param b_samples: Pre-generated MC samples for ancillary parameters.
        :param b_MC_steps: Number of MC steps for ancillary parameter sampling.
        """

        self.ancillary_obj = AncillaryParameter(
            b, u_b, corr_b, corr_between_b, b_samples, b_MC_steps
        )
        
    def build_from_obsarray(
        self,
        obs_ds: Any,
        y_name: str,
        measurement_func: Callable,
        initial_guess: Any,
        b_name: Optional[List[str]] = None,
        multiple_guess_measurements: bool = False,
        input_quantities_names: Union[str, List[str]] = None,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation: Optional[Any] = None,
        b_samples: Optional[Any] = None,
        b_MC_steps: Optional[int] = None,
    ) -> None:
        """
        Construct all retrieval input sub-objects from an ``obsarray`` dataset.

        The measurement variable, uncertainty, and error-correlation are read
        directly from the dataset.  Ancillary parameters are optionally
        sourced from the same dataset by name.

        :param obs_ds: ``obsarray`` dataset containing measurement and
            ancillary variables with associated uncertainty information.
        :param y_name: Name of the measurement variable in ``obs_ds``.
        :param measurement_func: Callable measurement/forward-model function.
        :param initial_guess: Initial values for the retrieval parameters.
        :param b_name: List of ancillary parameter variable names in
            ``obs_ds``, or ``None`` if no ancillary parameters are used.
        :param multiple_guess_measurements: If ``True``, the initial guess
            contains multiple measurements per parameter.
        :param input_quantities_names: Optional name(s) for input quantities.
        :param prior_shape: List of prior distribution shape names.
        :param prior_params: List of prior parameter dictionaries.
        :param prior_correlation: Correlation matrix for the prior.
        :param b_samples: Pre-generated MC samples for ancillary parameters.
        :param b_MC_steps: Number of MC steps for ancillary parameter sampling.
        """
        
        y = obs_ds[y_name].values
        u_y = obs_ds.unc[y_name].total_unc()
        corr_y = obs_ds.unc[y_name].total_err_corr_matrix()
        measurement_name = y_name
        
        if b_name is None:
            b = None
            u_b = None
            corr_b = None
            corr_between_b = None
            
        else:
            b = []
            u_b = []
            corr_b = []
            corr_between_b = None # corr between variables not yet implemented in obsarray, user could manually define after running function
            for name in b_name:
                b.append(obs_ds[name].values)
                u_b.append(obs_ds.unc[name].total_unc())
                corr_b.append(obs_ds.unc[name].total_err_corr_matrix())
        
        self.measurement_function_obj = MeasurementFunction(
            measurement_func,
            initial_guess,
            multiple_guess_measurements,
            measurement_name,
            input_quantities_names,
        )

        self.measurement_obj = Measurement(y, u_y, corr_y)

        self.ancillary_obj = AncillaryParameter(
            b, u_b, corr_b, corr_between_b, b_samples, b_MC_steps
        )
        self.prior_obj = Prior(prior_shape, prior_params, prior_correlation)