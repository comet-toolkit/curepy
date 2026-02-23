from curepy import (
    MeasurementFunction,
    Measurement,
    AncillaryParameter,
    RetrievalInput,
    Prior,
    MCMC,
    plot_corner,
)

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1503)

example_dir = os.path.dirname(os.path.abspath(__file__))


def quadratic(a, b, c, x, d):
    return a * x**2 + b * x + c + d


x = np.linspace(-5, 5, 200)
d = 5
data = 0.27 * x**2 - 0.19 * x - 8.5 + d
noise = np.random.normal(0, 1, data.shape)
y = data + noise

meas_func = MeasurementFunction(quadratic, [0.5, 0.2, -10])
meas = Measurement(y, noise, np.eye(len(x)))
ancill = AncillaryParameter([x, d], [None, 1], [np.eye(len(x)), None], b_MC_steps=3)
prior = Prior(
    ["normal"] * 3,
    [{"mu": 0.5, "sigma": 0.3}, {"mu": 0.1, "sigma": 0.05}, {"mu": 10, "sigma": 3}],
    prior_correlation="rand",
)

inputs = RetrievalInput(meas_func, meas, ancill, prior)

ret = MCMC(100, 1000, 100)

results = ret.run_retrieval(inputs, return_samples=True)

print(results.values)
print(results.uncertainties)
plt.plot(x, quadratic(*results.values, x, d))
plt.scatter(x, y, alpha=0.5, c="orange")
plt.savefig(os.path.join(example_dir, "test.png"))

corner = plot_corner(results.samples, labels=["a", "b", "c"])
corner.savefig(os.path.join(example_dir, "test_corner.png"))
