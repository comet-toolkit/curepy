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

Described below are the current retrieval methods implemented in **curepy**, ( #add info about future methods?)

Law of Propagation of Uncertainties (Optimal Estimation?) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Markov Chain Monte Carlo (MCMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

