from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pyprocar.cfg.base import BaseConfig, PlotType
from pyprocar.utils.plot_utils import (
    DEFAULT_COLORS,
    create_colormap,
    wes_anderson_palettes,
)


class FermiSurface2DMode(Enum):
    """
    An enumeration for defining the modes of 2D Fermi surface representations.

    Attributes
    ----------
    CONTOUR : str
        Represents the Fermi surface in a contour map.
    HEATMAP : str
        Represents the Fermi surface in a heatmap, indicating electron density.
    """
    PLAIN = "plain"
    PLAIN_BANDS = "plain_bands"
    PARAMETRIC = "parametric"
    SPIN_TEXTURE = "spin_texture"

class FermiSurface2DProperty(Enum):
    """
    An enumeration for defining the properties that can be visualized on the Fermi surface.

    Attributes
    ----------
    FERMI_SPEED : str
        Visualizes the scalar speed at which electrons travel at the Fermi level.
    FERMI_VELOCITY : str
        Visualizes the vector velocity of electrons moving at the Fermi level.
    HARMONIC_AVERAGE_EFFECTIVE_MASS : str
        Displays the harmonic average of the effective mass of electrons, reflecting the electron mobility.
    """
    FERMI_SPEED = "fermi_speed"
    FERMI_VELOCITY = "fermi_velocity"
    HARMONIC_AVERAGE_EFFECTIVE_MASS = "harmonic_effective_mass"


@dataclass
class FermiSurface2DConfig(BaseConfig):
    """
    Configuration class for plotting 2D Fermi surfaces with various customization options.

    Attributes
    ----------
    add_axes_labels : bool
        Flag to determine if axes labels should be added, default is True.
    add_legend : bool
        Flag to determine if a legend should be added, default is False.
    plot_color_bar : bool
        Flag to determine if a color bar should be plotted, default is False.
    cmap : str
        The colormap used for the plot, default is 'jet'.
    clim : Tuple[float, float]
        The color limits for the color bar.
    color : List[str]
        Colors for the spin plot lines.
    linestyle : List[str]
        Linestyles for the spin plot lines.
    linewidth : float
        The linewidth of the fermi surface, default is 0.2.
    no_arrow : bool
        Boolean to use no arrows to represent the spin texture, default is False.
    arrow_color : Optional[str]
        Color of the arrows if arrows are used.
    arrow_density : int
        The density of arrows in the spin texture, default is 10.
    arrow_size : int
        The size of the arrows in the spin texture, default is 3.
    spin_projection : str
        The projection used for the color scale in spin texture plots, default is 'z^2'.
    marker : str
        Controls the marker used in the spin plot, default is '.'.
    dpi : str
        The dpi value for saving images, uses the string 'figure' by default for dynamic assignment based on output type.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.

    Methods
    -------
    __post_init__():
        Post-initialization to set additional properties or default values.

    Examples
    --------
    To initialize a basic configuration with the default settings:

    >>> config = FermiSurface2DConfig()

    To customize the Fermi surface with a specific colormap and spin texture:

    >>> custom_config = FermiSurface2DConfig(cmap='magma', arrow_size=5)
    """
    modes: List[str] = field(default_factory=lambda: [mode.value for mode in FermiSurface2DMode])
    # Basic Plot Settings
    add_axes_labels: bool = True
    plot_color_bar: bool = False

    # Plot Appearance
    cmap: str = "jet"
    clim: Optional[Tuple[float, float]] = field(default_factory=lambda: (None, None))
    color: List[str] = field(default_factory=lambda: ['blue', 'red'])
    linestyle: List[str] = field(default_factory=lambda: ['solid', 'dashed'])
    linewidth: float = 0.2
    no_arrow: bool = False
    arrow_color: Optional[str] = None
    arrow_density: int = 10
    arrow_size: int = 3
    spin_projection: str = 'z^2'
    marker: str = '.'
    dpi: str = 'figure'


    def __post_init__(self):
        """This method is immediately called after the object is initialized. 
        It is useful to validate the data and set default values.
        """
        self.plot_type = PlotType.FERMI_SURFACE_2D

    def as_dict(self):
        """
        Returns a dictionary of the configuration settings.
        """
        return asdict(self)
