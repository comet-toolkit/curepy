"""Container for Measurement Function"""

from typing import Union, List, Callable, Optional, Any
import numpy as np
from copy import deepcopy


class MeasurementFunction:
    def __init__(
        self,
        measurement_func: Callable,
        initial_guess: Any,
        multiple_guess_measurements: bool = False,
        measurement_name: str = None,
        input_quantities_names: Union[str, List[str]] = None,
    ) -> None:
        """
        Container for the measurement (forward-model) function and the
        initial retrieval state.

        :param measurement_func: Callable measurement/forward-model function.
            Its positional arguments are the retrieval state-vector entries
            (in the same order as ``initial_guess``) optionally followed by
            ancillary parameters ``b``.
        :param initial_guess: Initial values for the retrieval parameters.
            May be a scalar, a 1-D iterable, or a 2-D iterable (one row per
            measurement location).
        :param multiple_guess_measurements: If ``True``, treat a 1-D
            ``initial_guess`` as a single row to be broadcast across multiple
            measurements.
        :param measurement_name: Optional name for the measured quantity.
        :param input_quantities_names: Optional name(s) for the retrieval
            input quantities.
        """

        self.measurement_function = measurement_func
        self._measurement_name = measurement_name
        self._input_quantities_names = input_quantities_names

        self.initial_guess = self._format_initial_guess(
            initial_guess, multiple_guess_measurements
        )

    @staticmethod
    def _format_initial_guess(
        initial_guess: Any,
        multiple_guess_measurements: bool = False,
    ) -> np.ndarray:
        """
        Format initial guess.

        :param initial_guess: Input initial guess for retrieval parameters.
        :param multiple_guess_measurements:  If ``True``, the initial guess
            contains multiple measurements per parameter.
        """
        # Handle nested case:
        if hasattr(initial_guess, "__len__") and hasattr(initial_guess[0], "__len__"):
            # If rectangular return array
            try:
                arr2d = np.array(initial_guess, dtype=float)
                if arr2d.ndim == 2:
                    return arr2d
            except Exception:
                pass
            # If ragged rows -> object array of 1-D float arrays
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

    def measurement_function_x(self, theta: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
        """
        Evaluate the measurement function at state vector ``theta``.

        Unpacks ``theta`` into the input-quantity tuple expected by the
        underlying measurement function and calls it, optionally passing
        ancillary parameters ``b``.

        :param theta: Flattened retrieval state vector.
        :param b: Ancillary parameter values, or ``None`` if not used.
        :returns: Output of the measurement function.
        """
        x = self.make_x_tuple(theta)
        if b is None:
            return self.measurement_function(*x)
        else:
            return self.measurement_function(*x, *b)

    def measurement_function_flattened_b(
        self,
        theta: np.ndarray,
        b_flat: np.ndarray,
        b_shape_list: List[tuple],
    ) -> np.ndarray:
        """
        Evaluate the measurement function with a flat ancillary-parameter vector.

        Reconstructs the original ancillary parameter arrays from the flat
        vector ``b_flat`` using the shapes in ``b_shape_list``, then calls
        the measurement function and returns a flattened result.

        :param theta: Flattened retrieval state vector.
        :param b_flat: Concatenated, flattened ancillary parameter values.
        :param b_shape_list: Shapes used to reconstruct each ancillary
            parameter array.
        :returns: Flattened output of the measurement function.
        """
        x = self.make_x_tuple(theta)
        num = 0
        b = np.empty(len(b_shape_list), dtype=object)
        for i, sh in enumerate(b_shape_list):
            num_sh = int(np.prod([x for x in sh]))
            b[i] = b_flat[num : num + num_sh].reshape(sh)
            num += num_sh

        return self.measurement_function(*x, *b).flatten()

    def measurement_function_flattened_output(
        self,
        theta: np.ndarray,
        b: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Evaluate the measurement function and return a flattened output array.

        :param theta: Flattened retrieval state vector.
        :param b: Ancillary parameter values, or ``None`` if not used.
        :returns: Flattened output of the measurement function.
        """
        x = self.make_x_tuple(theta)
        if b is None:
            out = self.measurement_function(*x)
        else:
            out = self.measurement_function(*x, *b)

        return out.flatten()

    def make_x_tuple(self, theta: np.ndarray) -> tuple:
        """
        Build the input-quantity tuple from the flattened state vector.

        Fills a deep copy of ``initial_guess`` with values from ``theta`` in
        order, supporting up to three levels of nesting.

        :param theta: Flattened state vector whose values are inserted into
            the initial-guess structure.
        :returns: Tuple of input quantities ready to be passed to the
            measurement function.
        """
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
