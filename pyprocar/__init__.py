from .version import (
    author as __author__,
    copyright as __copyright__,
    email as __email__,
    status as __status__,
    date as __date__,
)

from ._version import __version__

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

# TODO change all n* variables to n_* variable (norbital to n_orbital)
# TODO create a function in utils that does ProcarFileFilter
# TODO using * in import is not a good way, because 1. you don't know what's being imported, 2. the code looses it's organization @Logan
