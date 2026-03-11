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
The measurement function maps the state space to the measurement space. We know the meausrement function so it is easy to map the state space to the measurement space,
this is what we know as forward propagation. However, if we know the values of the measurements, :math:`y`, and want to calculate the values of the state vector, :math:`x`, 
we need a method of inverting :math:`f()`, and propagting uncertainties through the inversion process. 


Retrieval Methods
-----------------

Described below are the current retrieval methods implemented in **curepy**, all methods use the chi-square
function, :math:`\chi^{2}`, as a cost function where

..math::

    \chi^{2} = [f(x) - y]^{T}C_{y}^{-1}[f(x) - y]

The associated likelihood is defined as:

..math::

    P(y|x) = 2\pi^{\frac{n}{2}}|C_{y}^{\frac{-1}{2}}|\exp{\chi^{2}}



#add info about future methods?

Law of Propagation of Uncertainties (Optimal Estimation?) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Law of Propagation


Markov Chain Monte Carlo (MCMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

