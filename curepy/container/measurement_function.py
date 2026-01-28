"""Container for Measurement Function"""

from typing import Union, List

class MeasurementFunction:
    def __init__(self,
                 measurement_func,
                 initial_guess = None,
                 measurand_name: str = None,
                 input_quantities_names: Union[str, List[str]] = None,
                 ):
        
        self.measurement_func = measurement_func
        self._measurand_name = measurand_name
        self._input_quantities_names = input_quantities_names

        self.initial_guess = self.format_initial_guess(initial_guess)
        
    @staticmethod
    def _format_initial_guess(initial_guess):
        raise NotImplementedError
    
    def measurement_function_x(self, theta, b):
        x = self.make_x_tuple(theta)
        if b is not None:
            xb = x + tuple(b)
        else:
            xb = x
        return self.measurement_function(*xb)