"""Container for all retrieval inputs"""

from curepy.container.measurement_function import MeasurementFunction
from curepy.container.ancillary_parameter import AncillaryParameter
from curepy.container.prior import Prior
from curepy.container.measurement import Measurement

from typing import Union, List

class RetrievalInput:
    
    def __init__(self, 
                 measurement_function_obj: MeasurementFunction = None, 
                 measurement_obj: Measurement = None, 
                 ancillary_obj: AncillaryParameter = None, 
                 prior_obj: Prior = None,
    ):
        
        self.measurement_function_obj = measurement_function_obj
        self.measurement_obj = measurement_obj
        self.ancilary_obj = ancillary_obj
        self.prior_obj = prior_obj
        
    def build_retrieval_inputs(self,
                               measurement_func,
                                initial_guess,
                                y,
                                u_y = None,
                                corr_y = None,
                                multiple_guess_measurements: bool = False,
                                measurand_name: str = None,
                                input_quantities_names: Union[str, List[str]] = None,
                                prior_shape: str = None,
                                prior_params: dict = {},
                                b = None,
                                u_b = None,
                                corr_b = None,
                                corr_between_b = None,
                                b_samples = None,
                                b_iter = None,):
    
        self.measurement_function_obj = MeasurementFunction(measurement_func,
                 initial_guess,
                 multiple_guess_measurements,
                 measurand_name,
                 input_quantities_names)
        
        self.measurement_obj = Measurement(y, 
                                           u_y,
                                           corr_y)
        
        self.ancilary_obj = AncillaryParameter(b,
                                               u_b,
                                               corr_b,
                                               corr_between_b,
                                               b_samples,
                                               b_iter)
        self.prior_obj = Prior(prior_shape,
                               prior_params)
        
        
        
        