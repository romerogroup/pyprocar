from pyprocar import core, io, plotter, pyposcar, utils
from pyprocar._version import __version__
from pyprocar.io import *
from pyprocar.scripts import *
from pyprocar.utils import physics_constants, welcome
from pyprocar.utils.defaults import Settings
from pyprocar.utils.download_examples import download_from_hf, download_test_data
from pyprocar.utils.physics_constants import *
from pyprocar.version import author as __author__
from pyprocar.version import copyright as __copyright__
from pyprocar.version import date as __date__
from pyprocar.version import email as __email__
from pyprocar.version import status as __status__

# TODO change all n* variables to n_* variable (norbital to n_orbital)
# TODO create a function in utils that does ProcarFileFilter
# TODO using * in import is not a good way, because 1. you don't know what's being imported, 2. the code looses it's organization @Logan
