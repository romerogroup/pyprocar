from .version import (
    version as __version__,
    author as __author__,
    copyright as __copyright__,
    email as __email__,
    status as __status__,
    date as __date__,
)

import os

from .scripts import *
from .io import *
from . import io
from . import core
from . import utils
from . import plotter
from . import pyposcar
from .utils.download_examples import download_examples, download_example, download_dev_data
from .utils.defaults import Settings
from .utils import welcome

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TODO change all n* variables to n_* variable (norbital to n_orbital)
# TODO remove importing skimage, pyvista, trimesh, matplotlib, numpy, and seekpath. It is not needed here
# TODO create a function in utils that does ProcarFileFilter
# TODO create a directory named _old to move all of the old functionalities there. need to pass all of the old tests

# TODO using * in import is not a good way, because 1. you don't know what's being imported, 2. the code looses it's organization @Logan
