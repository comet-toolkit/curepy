===============
Getting Started
===============

**curepy** can be installed via pip::

    pip install curepy

The package can then be imported. The retrieval methods are located in the ``retrieval_methods`` module
and the input data containers are located in the ``container`` module::

    from curepy import retrieval_methods
    from curepy import container

Key concepts
------------

retrieval_methods
^^^^^^^^^^^^^^^^^

- Contains core retrieval algorithms such as optimal estimation (OE) and Markov Chain Monte Carlo (MCMC).
- Main class interfaces include ``OE`` and ``MCMC`` in ``curepy.retrieval_methods``.
- Each retrieval method requires a ``RetrievalInput`` object and returns a ``RetrievalResult`` containing:
  - retrieved state vector values,
  - uncertainties,
  - optional covariance/correlation and samples.

container module
^^^^^^^^^^^^^^^^

- Defines structured data classes used to build retrieval inputs:
  - ``Measurement`` (observations and uncertainty),
  - ``Prior`` (prior distribution settings),
  - ``AncillaryParameter`` (extra parameters with uncertainty/correlation),
  - ``MeasurementFunction`` (forward model and initial guess).
- Use these containers to assemble a consistent retrieval input stack before running a method.
- Facilitates separation of data definition (container) from retrieval execution (retrieval_methods).
