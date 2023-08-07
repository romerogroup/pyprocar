from . import vasp 
from . import qe
from . import lobster
from . import abinit
from . import siesta
from . import elk
# from . import bsxf
# from . import frmsf

from .parser import Parser

from .procarparser import ProcarParser
from .abinitparser import AbinitParser
from .elkparser import ElkParser
from .lobsterparser import LobsterParser, LobsterDOSParser, LobsterFermiParser
from .vaspxml import VaspXML


# from . import vaspxml

# TODO remove the directories abinitparser
# TODO remove the directories bxsfparser and change it to a *.py module
# TODO remove the directories elkparser and change it to a *.py module
# TODO remove the directories frmsfparser and change it to a *.py module
# TODO remove the directories lobsterparser and change it to a *.py module
# TODO remove the directories qeparser and change it to a *.py module
# TODO remove the directories vaspxml