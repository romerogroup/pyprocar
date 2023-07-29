# -*- coding: utf-8 -*-
from typing import Union

import os
import logging.config
from pathlib import Path
import yaml

from .utilsprocar import UtilsProcar
from .procarfilefilter import ProcarFileFilter
from .unfolder import Unfolder
from .splash import welcome
from . import mathematics
from . import defaults
from . import elements 
from . import sorting





# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # pyprocar
LOGGING_NAME = 'pyprocar'
VERBOSE = str(os.getenv('PYPROCAR_VERBOSE', True)).lower() == 'true'  # global verbose mode


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})
    

# Set logger
set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

