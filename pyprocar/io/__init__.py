from . import vasp 
from . import qe
from . import lobster

from .procarparser import ProcarParser
from .frmsfparser import FrmsfParser
from .bxsfparser import BxsfParser
from .abinitparser import AbinitParser
from .elkparser import ElkParser
from .qeparser import QEParser, QEDOSParser, QEFermiParser
from .lobsterparser import LobsterParser, LobsterDOSParser, LobsterFermiParser
# from .lobsterparser import LobsterDOSParser, LobsterFermiParser
from .vaspxml import VaspXML
# from . import vaspxml

# TODO remove the directories abinitparser
# TODO remove the directories bxsfparser and change it to a *.py module
# TODO remove the directories elkparser and change it to a *.py module
# TODO remove the directories frmsfparser and change it to a *.py module
# TODO remove the directories lobsterparser and change it to a *.py module
# TODO remove the directories qeparser and change it to a *.py module
# TODO remove the directories vaspxml