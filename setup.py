# -*- coding: utf-8 -*-
'''
pyOptimalEstimation

Copyright (C) 2014-15 Maximilian Maahn, IGMK (mmaahn_(AT)_meteo.uni-koeln.de)
http://gop.meteo.uni-koeln.de/software

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

from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":
  setup(
      name='pyOptimalEstimation',
      version='0.2dev',
      packages=['pyOptimalEstimation',],
      license='GNU General Public License 3',
      author="Maximilian Maahn",
      author_email="maximilian.maahn@colorado.edu",
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          ],
      install_requires = ['numpy', 'matplotlib', 'pandas' , 'scipy'],
      long_description=long_description,
      long_description_content_type='text/markdown'
      )