"""Container for Measurement Function"""

from typing import Union, List
import numpy as np
from copy import deepcopy

class MeasurementFunction:
    def __init__(self,
                 measurement_func,
                 initial_guess,
                 multiple_guess_measurements: bool = False,
                 measurand_name: str = None,
                 input_quantities_names: Union[str, List[str]] = None,
                 ):
        
        self.measurement_function = measurement_func
        self._measurand_name = measurand_name
        self._input_quantities_names = input_quantities_names
    
        self.initial_guess = self._format_initial_guess(initial_guess, multiple_guess_measurements)
        
    @staticmethod
    def _format_initial_guess(
        initial_guess,
        multiple_guess_measurements: bool = False,
    ):
        """
        Format initial guess
        
        :param initial_guess: Input initial guess for retrieval parameters
        :param multiple_guess_measurements: Bool, True if initial guess contains multiple measurements for each parameter else False

        """
        # Handle nested case:
        if hasattr(initial_guess, "__len__") and hasattr(initial_guess[0], "__len__"):
            #If rectangular return array
            try:
                arr2d = np.array(initial_guess, dtype=float)
                if arr2d.ndim == 2:
                    return arr2d
            except Exception:
                pass
            #If ragged rows -> object array of 1-D float arrays
            ig_obj = np.empty(len(initial_guess), dtype=object)
            for i, row in enumerate(initial_guess):
                ig_obj[i] = np.array(row, dtype=float)
            return ig_obj

        # At this point, it is scalar-like or 1-D iterable
        if not hasattr(initial_guess, "__len__"):
            # Scalar
            return np.array([initial_guess], dtype=float)

        arr = np.array(initial_guess, dtype=float)
        # arr is 1-D
        if multiple_guess_measurements:
            return arr[None, :]
        else:
            return arr

    def measurement_function_x(self, theta, b):
        x = self.make_x_tuple(theta)
        if b is None:
            return self.measurement_function(*x) #todo: tests for edge cases
        else:
            return self.measurement_function(*x, *b)
          
    def make_x_tuple(self, theta):
        x = deepcopy(self.initial_guess)
        j = 0
        for i in range(len(x)):
            if not hasattr(x[i], "__len__"):
                x[i] = theta[j]
                j += 1
            else:
                for ii in range(len(x[i])):
                    if not hasattr(x[i][ii], "__len__"):
                        x[i][ii] = theta[j]
                        j += 1
                    else:
                        for iii in range(len(x[i][ii])):
                            if not hasattr(x[i][ii][iii], "__len__"):
                                x[i][ii][iii] = theta[j]
                                j += 1
                            else:
                                raise ValueError(
                                    "The initial guess has too high dimensionality."
                                )
        return tuple(x)