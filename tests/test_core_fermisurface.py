import os
import pytest
import numpy as np
from typing import List
from scipy.interpolate import CubicSpline
import networkx as nx
from matplotlib import pylab as plt
import pyvista
from pyprocar.core import ElectronicBandStructure
from pyprocar.io import Parser
import pyprocar
# Helper function to create a dummy ElectronicBandStructure instance

DATA_DIR = f"{pyprocar.PROJECT_DIR}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}fermi"


