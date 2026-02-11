from curepy.container.measurement_function import MeasurementFunction
from curepy.container.measurement import Measurement
from curepy.container.ancillary_parameter import AncillaryParameter
from curepy.container.prior import Prior
from curepy.retrieval_methods.LPU import LPU
from curepy.container.retrieval_input import RetrievalInput
from curepy.utilities.plotting import corner

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1503)

def quadratic(a, b, c, x, d):
    return a*x**2 + b*x + c + d

x = np.linspace(-5, 5, 200)
d = 5
data = 0.27*x**2 -0.19*x -8.5
noise = np.random.normal(0, 1, data.shape)
y = data + noise

meas_func = MeasurementFunction(quadratic, [0.5, 0.2, -10])
meas = Measurement(y, noise, np.eye(len(x)))
ancill = AncillaryParameter([x, d], [0.01*np.ones_like(x), 0.05], [np.eye(len(x)), np.array([1,])])
prior = Prior(["normal"]*3, [{"mu": 0.5, "sigma": 0.3}, 
                        {"mu": 0.1, "sigma": 0.05},
                        {"mu": 10, "sigma": 3}],
              prior_correlation="rand")

inputs = RetrievalInput(meas_func, meas, ancill, prior)

ret = LPU()

results = ret.run_retrieval(inputs)

print(results.values)
print(results.uncertainties)
plt.plot(x, quadratic(*results.values, x, d))
plt.scatter(x, y, alpha = 0.5, c = 'orange')
plt.savefig("C:/Users/jr20/code/curepy/examples/LPU_test.png")
