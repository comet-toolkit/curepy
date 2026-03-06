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