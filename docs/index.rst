Welcome to curepy's documentation!
=========================================================

The **curepy** module is a Python software package to propagate uncertainties through inverse problems. 

**curepy** can be used to solve inverse problems with input measurements and uncertainties inputted manually, or using an **obsarray** dataset.
It can also be used to analyse the outputs of these inverse problems to quantify uncertainties and compare results.
This documentation provides general information on how to use the module (with some examples), as well as a detailed API of the included classes and function.

.. grid:: 2
    :gutter: 2

    .. grid-item-card::  Getting Started
        :link: content/getting_started
        :link-type: doc

        New to *curepy*? Check out the getting started guides. They contain installation instructions.

    .. grid-item-card::  ATBD
        :link: content/atbd
        :link-type: doc

        The theory section provides a mathematical basis to inverse problems, and a description of the retrieval methods included in *curepy*.

    .. grid-item-card::  User Guide
        :link: content/user_guide
        :link-type: doc

        The user guide provides in-depth information on how to use the retrieval methods in *curepy*.


    .. grid-item-card::  API reference
        :link: content/api/curepy
        :link-type: doc

        The reference guide contains a detailed description of the *curepy* API.
        The reference describes how the methods work and which parameters can
        be used. It assumes that you have an understanding of the key concepts.

.. toctree::
   :maxdepth: 1
   :caption: For users:
   :hidden:

   Getting Started <content/getting_started>
   ATBD <content/atbd>
   User Guide <content/user_guide>
   API Reference <content/api/curepy>