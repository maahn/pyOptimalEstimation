:mod:`pyOptimalEstimation` Package
==================================


.. toctree::
   :maxdepth: 3

Python package to solve an inverse problem using Optimal Estimation and an arbitrary Forward model following Rodgers, 2000.


Download
--------

The code is available at https://github.com/maahn/pyOptimalEstimation


Reference
---------

You find more information about pyOptimalEstimation and examples in:

Maahn, M., D. D. Turner, U. Löhnert, D. J. Posselt, K. Ebell, G. G. Mace, and J. M. Comstock, 2020: Optimal Estimation Retrievals and Their Uncertainties: What Every Atmospheric Scientist Should Know. Bull. Amer. Meteor. Soc., doi:https://doi.org/10.1175/BAMS-D-19-0027.1

Please reference to our publication if you use the pyOptimalEstimation package

Examples
--------

Please see pyOptimalEstimation/examples for a minimal working example. For more extensive, interactive examples, check out  https://github.com/maahn/pyOptimalEstimation_examples and our paper.


Installation
------------
Make sure you use Python 3.10 or newer.

Change to the folder containing the project and do ::

  python -m pip install .

in the terminal. If you do not have root privileges, you can also do ::

  python -m pip install --user .

which will install pyOptimalEstimation in userbase/lib/pythonX.Y/site-packages.


API documentation
-----------------

.. automodule:: pyOptimalEstimation.pyOEcore
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
