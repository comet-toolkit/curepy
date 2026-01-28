"""Container for Measurement Function"""

from typing import Union, List
import numpy as np

class MeasurementFunction:
    def __init__(self,
                 measurement_func,
                 initial_guess,
                 multiple_guess_measurements: bool = False,
                 measurand_name: str = None,
                 input_quantities_names: Union[str, List[str]] = None,
                 ):
        
        self.measurement_func = measurement_func
        self._measurand_name = measurand_name
        self._input_quantities_names = input_quantities_names
    
        self.initial_guess = self.format_initial_guess(initial_guess, multiple_guess_measurements)
        
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

        # 1-D iterable (non-empty or empty)
        try:
            arr = np.array(initial_guess, dtype=float)
        except Exception as e:
            raise TypeError(f"initial_guess must be numeric or nested numeric sequences; got {type(initial_guess)}") from e

        # arr is 1-D
        if multiple_guess_measurements:
            return arr[None, :]
        else:
            return arr

    def measurement_function_x(self, theta, b):
        x = self.make_x_tuple(theta)
        if b is not None:
            xb = x + tuple(b)
        else:
            xb = x
        return self.measurement_function(*xb)