# -*- coding: utf-8 -*-
'''
pyOptimalEstimation

# Copyright (C) 2014-21 Maximilian Maahn, Leipzig University
# maximilian.maahn@uni-leipzig.de
# https://github.com/maahn/pyOptimalEstimation

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .pyOEcore import optimalEstimation, optimalEstimation_loadResults, invertMatrix

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError: # for Pyton 3.6 and 3.7
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pyOptimalEstimation")
except PackageNotFoundError:
    # package is not installed
    pass