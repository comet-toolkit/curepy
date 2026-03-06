=====
Usage
=====

Getting Started
------

``curepy`` can be installed via pip::

    pip install curepy

The package can then be imported. The retrieval methods are located in the ``retrieval_methods`` module
and the input data containers are located in the ``container`` module::

    from curepy import retrieval_methods
    from curepy import container

Instantiating a Retrieval Input
-------------------------------

Every retrieval method requires a ``RetrievalInput`` object to run a retrieval::

    from curepy.retrieval_methods.retrieval_input import RetrievalInput
    ret = RetrievalInput()

This object can be instatiated using a combination of containers or by using the ``RetrievalInput().build_retrieval_inputs()`` function.
Individual containers can also be built within the ``RetrievalInput`` object using the individual 'build' functions::

    ret.build_measurement()
    ret.build_prior()
    ret.build_ancillary()
    ret.build_measurement_function()

These functions build the ``Measurement``, ``Prior``, ``AncillaryParameter``, and ``MeasurementFunction`` objects, respectively.

Every retrieval method requires a ``Measurement`` and ``MeasurementFunction`` input to be set in the ``RetrievalInput``.
``Prior`` and ``AncillaryParameter`` objects are optional.

Measurement
^^^^^^^^^^^

The ``Measurement`` object stores the measurements, :math:`y`, and any related uncertainty and correlation information. 

MeasurementFunction
^^^^^^^^^^^^^^^^^^^

The ``MeasurementFunction`` object stores the measurement function, :math:`f()` and an initial guess for the values of :math:`x`.
There are is also an optional Boolean input `multiple_guess_measurements`, if False, the initial guess input is a valid input to the measurement function,
if True, the initial guess input is made up of multiple valid inputs to the measurement function joined along the first dimension. By default, this is set to False.

Prior
^^^^^

The ``Prior`` object stores information used to define the prior distribution. The inputs are `prior_shape`, a List of the shapes of each prior, `prior_params`,
a List of Dictionaries of each priors' parameters, and an optional `prior_correlation`, a correlation matrix describing the correlation between each prior distribution.
The length of `prior_shape`, `prior_params`, and the side length of `prior_correlation` must be equal to the number of components in :math:`\underline{x}`. 

.. note::
    If `prior_correlation` is not defined, it is set to random as default (the identity matrix).

The table of valid prior shapes and associated parameters can be found below

+--------------+-----------------------------------+
| Shape        | Parameters                        |
+==============+===================================+
| uniform      | * minimum                         |
|              | * maximum                         |
+--------------+-----------------------------------+
| normal       | * mu (mean)                       |
|              | * sigma (standard deviation)      |
+--------------+-----------------------------------+