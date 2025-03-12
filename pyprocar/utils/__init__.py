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
from pyprocar.utils.log_utils import setup_logging

# Initialize logger
setup_logging()