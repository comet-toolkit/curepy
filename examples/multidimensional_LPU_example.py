from curepy import (
    MeasurementFunction,
    Measurement,
    AncillaryParameter,
    LPU,
    RetrievalInput,
)

import os
import numpy as np

np.random.seed(1503)
example_dir = os.path.dirname(os.path.abspath(__file__))


def quadratic(a, b, c, x, d):
    return a * x**2 + b * x + c + d


x = np.ones((5, 5)) * np.linspace(-5, 5, 5)
d = 5
data = 0.27 * x**2 - 0.19 * x - 8.5 + d
noise = np.random.normal(0, 1, data.shape)
y = data + noise

meas_func = MeasurementFunction(quadratic, [0.5, 0.2, -10])
meas = Measurement(y, noise, "rand")
ancill = AncillaryParameter(
    [x, d],
    [0.01 * np.ones_like(x), 1],
    [
        np.eye(len(x.flatten())),
        np.array(
            [
                1,
            ]
        ),
    ],
    b_MC_steps=10,
)

inputs = RetrievalInput(meas_func, meas, ancill)

ret = LPU()

results = ret.run_retrieval(inputs)

print(results.values)
print(results.uncertainties)
