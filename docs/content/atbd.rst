===========================
Algorithm Theoretical Basis
===========================

Inverse Problems
----------------

Consider a measurement function :math:`f()`, where,

.. math::

    y = f(\underline{x}, \underline{b})

y is a variable that can be measured, x is a vector of parameters that we want to retrieve (sometimes called the state vector), 
and b is a vector of inputs to the measurement function that can be measured (sometimes called the ancillary vector).
The measurement function maps the state space to the measurement space. We know the measurement function so it is easy to map the state space to the measurement space,
this is what we know as forward propagation. However, if we know the values of the measurements, :math:`y`, and want to calculate the values of the state vector, :math:`x`, 
we need a method of inverting :math:`f()`, and propagting uncertainties through the inversion process. 


Retrieval Methods
-----------------

Described below are the current retrieval methods implemented in **curepy**, all methods use the chi-square
function, :math:`\chi^{2}`, as a cost function where

.. math::

    \chi^{2} = [f(x) - y]^{T}C_{y}^{-1}[f(x) - y] + [x - x_{a}]^{T}C_{a}^{-1}[x - x_{a}]

where:

* :math:`C_{y}` is the measurement error covariance matrix, representing uncertainties and correlations in the measurements.

* :math:`x_{a}` is the a priori state vector, a best guess or physically reasonable estimate of :math:x before considering the measurements.

* :math:`C_{a}` is the a priori covariance matrix, expressing uncertainty in the prior knowledge of the state vector.

The prior distribution encodes what we know (or assume) about the state vector before considering the measurements.
Priors are essential because multiple states can produce nearly identical measurements,
the measurements may be noisy, and the forward model may have limited sensitivity to certain components of the state vector.
The prior stabilizes the inversion and helps guide the solution toward physically meaningful values.

The associated likelihood is defined as:

.. math::

    P(y|x) = 2\pi^{\frac{n}{2}}|C_{y}^{-\frac{1}{2}}|e^{-\frac{1}{2} \chi^{2}}

where :math:`n` is the dimensionality of the measurement vector. 

Optimal Estimation (OE)
^^^^^^^^^^^^^^^^^^^^^^^

The Optimal Estimation method is described thoroughly in [1]_. It involves minimising the cost function to find the optimal values of the state vector, 
then the associated uncertainties and correlations are calculated from the posterior covariance, :math:`C_{x}` which itself is calculated using the Jacobian of the measurement function with respect to :math:`x, :math:`J`,

.. math::

    C_{x} = (J^{T}C_{\eta}J + C_{a}^{-1})^{-1}

    C_{\eta} = C_{y} + J_{b}C_{b}J_{b}^{T}

where:

* :math:`J_{b}` is the Jacobian of the measurement function with respect to the ancillary vector.

* :math:`C_{b}` is the ancillary vector covariance matrix.


.. [1] https://www.sciencedirect.com/science/article/pii/S0034425718303304

Markov Chain Monte Carlo (MCMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Markov Chain Monte Carlo method approximates the posterior distribution  by generating a sequence (a chain) of correlated samples.
Unlike optimization-based retrievals that return a single “best-fit” solution, MCMC retrievals provide the full posterior distribution,
allowing non-Gaussian uncertainties, multimodal solutions, and parameter correlations to be characterized naturally.

*curepy* implements MCMC using the Metropolis-Hastings algorithm:

* An initial guess :math:`x_{0}` is chosen.
* A move is proposed to :math:`x_{i}` using MC sampling.
* The move is evaluated by comparing the likelihood of the new state compared to the initial state based on the cost function. 
* The move is then accepted or rejected depending on this likelihood.
* This is repeated for N steps.
* The initial samples are discarded so the chain 'forgets' its initial position.
* The remaining samples then approximate the posterior distribution.


Corner plots and trace plots can be used to assess the extent to which the posterior distribution has been reached. Trace plots show the evolution of each parameter through the chain,
a well-mixed chain has no long-term drift. Corner plots display parameter posterior distributions and correlation between parameters.
