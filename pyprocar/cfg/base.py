
from typing import Dict, Any
from enum import Enum, auto
from dataclasses import dataclass, field

class PlotType(Enum):
    """
    Enumeration for specifying the type of plot.

    Attributes
    ----------
    FERMI_SURFACE_3D : str
        Represents a 3D Fermi surface plot.
    FERMI_SURFACE_2D : str
        Represents a 2D Fermi surface plot.
    BAND_STRUCTURE : str
        Represents a band structure plot.
    DENSITY_OF_STATES : str
        Represents a density of states plot.

    These identifiers are used to select the appropriate plotting configuration
    and behavior within the application.
    """
    FERMI_SURFACE_3D = 'fermi_surface_3d'
    FERMI_SURFACE_2D = 'fermi_surface_2d'
    BAND_STRUCTURE = 'band_structure'
    BAND_STRUCTURE_2D = 'band_structure_2d'
    DENSITY_OF_STATES = 'density_of_states'
    UNFOLD = 'unfold'


@dataclass
class BaseConfig:
    """
    Base class for configuration settings of various plot types.

    This class serves as a foundation for more specialized configuration classes,
    ensuring a consistent interface across different plot configurations.

    Attributes
    ----------
    plot_type : PlotType
        The type of plot this configuration is meant for. It dictates the
        specific configuration subclass and related plotting functionalities.

    custom_settings : Dict[str, Any]
        A dictionary to hold any additional settings that do not fit directly
        into the static structure of a configuration class. This provides flexibility
        in adding new settings without altering the class structure.

    Example
    -------
    Creating an instance of BaseConfig for a 3D Fermi surface plot:

    >>> config = BaseConfig(plot_type=PlotType.FERMI_SURFACE_3D)
    >>> print(config.plot_type)
    PlotType.FERMI_SURFACE_3D
    """
    plot_type: PlotType
    custom_settings: Dict[str, Any] = field(default_factory=dict)