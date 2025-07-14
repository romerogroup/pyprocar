# -*- coding: utf-8 -*-
from typing import Union

# Loading configuration settings
from pyprocar.utils.config import (
    CONFIG,
    DATA_DIR,
    LOG_DIR,
    PKG_DIR,
    ROOT,
    ConfigManager,
)
from pyprocar.utils.log_utils import setup_logging

# from pyprocar.utils.procarfilefilter import ProcarFileFilter
from pyprocar.utils.splash import welcome

# from pyprocar.utils import defaults, elements, math, sorting, strings



# from pyprocar.utils.utilsprocar import UtilsProcar

# Initialize logger
setup_logging()
