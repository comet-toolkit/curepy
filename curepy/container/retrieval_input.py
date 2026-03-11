"""Container for all retrieval inputs"""

from curepy.container.measurement_function import MeasurementFunction
from curepy.container.ancillary_parameter import AncillaryParameter
from curepy.container.prior import Prior
from curepy.container.measurement import Measurement

from typing import Union, List


class RetrievalInput:

    def __init__(
        self,
        measurement_function_obj: MeasurementFunction = None,
        measurement_obj: Measurement = None,
        ancillary_obj: AncillaryParameter = None,
        prior_obj: Prior = None,
    ):

        self.measurement_function_obj = measurement_function_obj
        self.measurement_obj = measurement_obj
        self.ancillary_obj = ancillary_obj
        self.prior_obj = prior_obj

    def build_retrieval_inputs(
        self,
        measurement_func,
        initial_guess,
        y,
        u_y=None,
        corr_y=None,
        multiple_guess_measurements: bool = False,
        measurement_name: str = None,
        input_quantities_names: Union[str, List[str]] = None,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation=None,
        b=None,
        u_b=None,
        corr_b=None,
        corr_between_b=None,
        b_samples=None,
        b_MC_steps=None,
    ):

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

    def build_measurement_function(
        self,
        measurement_func,
        initial_guess,
        multiple_guess_measurements: bool = False,
        measurement_name: str = None,
        input_quantities_names: Union[str, List[str]] = None,
    ):

        self.measurement_function_obj = MeasurementFunction(
            measurement_func,
            initial_guess,
            multiple_guess_measurements,
            measurement_name,
            input_quantities_names,
        )

    def build_measurement(
        self,
        y,
        u_y=None,
        corr_y=None,
    ):

        self.measurement_obj = Measurement(y, u_y, corr_y)

    def build_prior(
        self,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation=None,
    ):

        self.prior_obj = Prior(prior_shape, prior_params, prior_correlation)

    def build_ancillary(
        self,
        b=None,
        u_b=None,
        corr_b=None,
        corr_between_b=None,
        b_samples=None,
        b_MC_steps=None,
    ):

        self.ancillary_obj = AncillaryParameter(
            b, u_b, corr_b, corr_between_b, b_samples, b_MC_steps
        )
        
    def build_from_obsarray(
        self,
        obs_ds,
        y_name: str,
        measurement_func,
        initial_guess,
        b_name: List[str] = None,
        multiple_guess_measurements: bool = False,
        input_quantities_names: Union[str, List[str]] = None,
        prior_shape: List[str] = None,
        prior_params: List[dict] = [{}],
        prior_correlation=None,
        b_samples=None,
        b_MC_steps=None,        
    ):
        
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
                corr_b.append(obs_ds.unc[name].total_err_corr_matrix())#todo: add handling if no uncertainty
        
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