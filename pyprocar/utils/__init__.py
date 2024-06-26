# -*- coding: utf-8 -*-
from typing import Union



from pyprocar.utils.utilsprocar import UtilsProcar
from pyprocar.utils.procarfilefilter import ProcarFileFilter
from pyprocar.utils.splash import welcome
from pyprocar.utils import mathematics
from pyprocar.utils import defaults
from pyprocar.utils import elements 
from pyprocar.utils import sorting
from pyprocar.utils import strings

# Loading configuration settings
from pyprocar.utils.config import ConfigManager
from pyprocar.utils.config import LOG_DIR,DATA_DIR,ROOT,PKG_DIR,CONFIG
from pyprocar.utils.log_config import setup_logging

# Initialize logger
LOGGER = setup_logging(log_dir=LOG_DIR,apply_filter=CONFIG['APPLY_LOG_FILTER'], log_level=CONFIG['log_level'])