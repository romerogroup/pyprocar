# Copyright (C) 2020 Francisco Munoz, Aldo Romero, Sobhit Singh, Uthpala Herath, Pedram Tavadze, Eric Bousquet, Xu He, Reese Boucher, Logan Lang, Freddy Farah
#
# This file is part of PyProcar.
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

from .version import version as __version__
from .version import author as __author__
from .version import copyright as __copyright__
from .version import email as __email__
from .version import status as __status__
from .version import date as __date__


import logging
import re
import sys
import os

# import ase
import matplotlib.pyplot as plt
import numpy as np
import seekpath
import skimage
# import pychemia
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
from pyprocar.lobsterparser import LobsterParser, LobsterDOSParser, LobsterFermiParser

from pyprocar.fermisurface import FermiSurface
from pyprocar.fermisurface3d import FermiSurface3D, BrillouinZone

# from . import fermisurface3d
from pyprocar.procarfilefilter import ProcarFileFilter
from pyprocar.procarparser import ProcarParser
from pyprocar.procarplot import ProcarPlot
from pyprocar.procarselect import ProcarSelect
from pyprocar.procarsymmetry import ProcarSymmetry
from pyprocar.utilsprocar import UtilsProcar
from pyprocar.procarunfold import ProcarUnfolder
from pyprocar import doscarplot
from pyprocar import vaspxml

from .splash import welcome

from .scripts import *
from . import io
from . import core
from . import utils
from . import plotter
from .utils.defaults import Settings


