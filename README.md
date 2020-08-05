[![PyPI version](https://badge.fury.io/py/pyOptimalEstimation.svg)](https://badge.fury.io/py/pyOptimalEstimation)
[![Documentation Status](https://readthedocs.org/projects/pyoptimalestimation/badge/?version=latest)](https://pyoptimalestimation.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://www.travis-ci.org/maahn/pyOptimalEstimation.svg?branch=master)](https://www.travis-ci.org/maahn/pyOptimalEstimation)
[![Coverage Status](https://coveralls.io/repos/github/maahn/pyOptimalEstimation/badge.svg?branch=master)](https://coveralls.io/github/maahn/pyOptimalEstimation?branch=master)
[![Downloads](https://pepy.tech/badge/pyoptimalestimation)](https://pepy.tech/project/pyoptimalestimation)

# "pyOptimalEstimation" Package v1.1

Python package to solve an inverse problem using Optimal Estimation
and an arbitrary Forward model following Rodgers, 2000.

## Installation

Change to the folder containing the project and do 
```
  python setup.py install
```
in the terminal. If you do not have root privileges, you can also do:
```
  python setup.py --user install
```
which will install pyOptimalEstimation in `userbase/lib/pythonX.Y/site-packages`
or
```
  python setup.py install --home=~
```
which will install pyOptimalEstimation in `~/lib/python`.

## Reference

Please reference to our paper if you use the pyOptimalEstimation package

Maahn, M., D. D. Turner, U. Löhnert, D. J. Posselt, K. Ebell, G. G. Mace, and J. M. Comstock, 2020: Optimal Estimation Retrievals and Their Uncertainties: What Every Atmospheric Scientist Should Know. Bull. Amer. Meteor. Soc., doi:https://doi.org/10.1175/BAMS-D-19-0027.1

## Examples

Annotated example applications are available at https://github.com/maahn/pyOptimalEstimation_examples.

## API documentation

See https://pyoptimalestimation.readthedocs.io/en/latest/ for documentation.

