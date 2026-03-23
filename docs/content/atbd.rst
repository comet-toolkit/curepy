===========================
Algorithm Theoretical Basis
===========================

Inverse Problems
----------------

Consider a measurement function :math:`f()`, where,

.. math::

    y = f(\underline{x}, \underline{b})

y is a variable that can be measured, x is a vector of parameters that we want to retrieve (sometimes called the state vector), 
and b is a vector of inputs to the measurement function that can be measured.
The measurement function maps the state space to the measurement space. We know the measurement function so it is easy to map the state space to the measurement space,
this is what we know as forward propagation. However, if we know the values of the measurements, :math:`y`, and want to calculate the values of the state vector, :math:`x`, 
we need a method of inverting :math:`f()`, and propagting uncertainties through the inversion process. 


Retrieval Methods
-----------------

Described below are the current retrieval methods implemented in **curepy**, all methods use the chi-square
function, :math:`\chi^{2}`, as a cost function where

.. math::

    \chi^{2} = [f(x) - y]^{T}C_{y}^{-1}[f(x) - y] + [x - x_{a}]^{T}C_{a}^{-1}[x - x_{a}]

The associated likelihood is defined as:

.. math::

    P(y|x) = 2\pi^{\frac{n}{2}}|C_{y}^{\frac{-1}{2}}|\exp{\chi^{2}}


Optimal Estimation (OE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Optimal Estimation method is described thoroughly in [1]. It involves minimising the cost function to find the optimal values of the state vector, 
then the associated uncertainties and correlations are calculated from the posterior covariance which itself is calculated using the Jacobian.

.. [1] https://www.sciencedirect.com/science/article/pii/S0034425718303304

Markov Chain Monte Carlo (MCMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Markov Chain Monte Carlo method approximates the posterior distribution by sampling the state vector space. An initial guess :math:`x_{0}` is chosen then a move is proposed to :math:`x_{i}` using MC sampling.
The move is evaluated by comparing the likelihood of the new state compared to the initial state based on the cost function. The move is then accepted or rejected depending on this likelihood.
This is repeated for N steps, and the initial samples are discarded so the chain of samples 'forgets' its initial position. The remaining samples then approximate the posterior distribution.

Corner plots and trace plots can be used to assess the extent to which the posterior distribution has been reached.
