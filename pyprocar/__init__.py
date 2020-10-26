__author__ = "Francisco Munoz,Aldo Romero,Sobhit Singh,Uthpala Herath,Pedram Tavadze,Eric Bousquet,Xu He"
__copyright__ = "Copyright 2020"
__version__ = "5.5.6"
__email__ = (
    "fvmunoz@gmail.com/alromero@mail.wvu.edu/ukh0001@mix.wvu.edu/petavazohi@mix.wvu.edu"
)
__status__ = "Development"
__date__ = "Oct 26th, 2020"

# Copyright (C) 2020 Francisco Munoz,Aldo Romero,Sobhit Singh,Uthpala Herath,Pedram Tavadze,Eric Bousquet,Xu He
#
# This file is part of pyprocar.
#
# PyProcar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyProcar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pyprocar.  If not, see <http://www.gnu.org/licenses/>.

import logging
import re
import sys

import ase
import matplotlib.pyplot as plt
import numpy as np
import seekpath
import skimage
import pychemia
import pyvista
import trimesh

# Frmsf
from pyprocar.frmsfparser import FrmsfParser

# Bxsf
from pyprocar.bxsfparser import BxsfParser

# Abinit
from pyprocar.abinitparser import AbinitParser

# Elk
from pyprocar.elkparser import ElkParser

# Quantum Espresso
from pyprocar.qeparser import QEParser, QEDOSParser, QEFermiParser

# Lobster
from pyprocar.lobsterparser import LobsterParser, LobsterDOSParser

from pyprocar.fermisurface import FermiSurface
from pyprocar.fermisurface3d import FermiSurface3D

# from . import fermisurface3d
from pyprocar.procarfilefilter import ProcarFileFilter
from pyprocar.procarparser import ProcarParser
from pyprocar.procarplot import ProcarPlot
from pyprocar.procarselect import ProcarSelect
from pyprocar.procarsymmetry import ProcarSymmetry
from pyprocar.utilsprocar import UtilsProcar
from pyprocar import doscarplot
from pyprocar import vaspxml
from pyprocar import core

from .scriptBandsDosplot import bandsdosplot
from .scriptBandsplot import bandsplot
from .scriptCat import cat
from .scriptDosplot import dosplot
from .scriptFermi2D import fermi2D
from .scriptFermi3D import fermi3D

# from .scriptSpin_asymmetry import spin_asymmetry
from .scriptFilter import filter
from .scriptKmesh2D import generate2dkmesh
from .scriptKpath import kpath
from .scriptRepair import repair
from .scriptUnfold import unfold
from .scriptVector import Vector
from .scriptBandGap import bandgap
from .splash import welcome
