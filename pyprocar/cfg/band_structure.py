from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pyprocar.cfg.base import BaseConfig, PlotType


class BandStructureMode(Enum):
    """
    An enumeration for defining the modes of Band Structure representations.

    Attributes
    ----------
    PLAIN : str
        Represents the band structure in a simple, where the colors are the different bands.
    PARAMETRIC : str
        Represents the band structure in a parametric form, summing over the projections.
    SACATTER : str
        Represents the band structure in a scatter plot, where the colors are the different bands.
    ATOMIC : str
        Represents the band structure in an atomic level plot, plots singlr kpoint bands.
    OVERLAY : str
        Represents the band structure in an overlay plot, where the colors are the selected projections
    OVERLAY_SPECIES : str
        Represents the band structure in an overlay plot, where the colors are 
        the different projection of the species.
    OVERLAY_ORBITALS : str
        Represents the band structure in an overlay plot, where  the colors are 
        the different projection of the orbitals.
    """
    PLAIN = "plain"
    PARAMETRIC = "parametric"
    SACATTER = "scatter"
    ATOMIC = "atomic"
    OVERLAY = "overlay"
    OVERLAY_SPECIES = "overlay_species"
    OVERLAY_ORBITALS = "overlay_orbitals"


class BandStructureProperty(Enum):
    """
    An enumeration for defining the properties that can be visualized on the BandStructure.

    Attributes
    ----------
    """


@dataclass
class BandStructureConfig(BaseConfig):
    """
    Configuration class for plotting band structures with custom options.

    Parameters
    ----------
    color : str, optional (default 'black')
        Color for the plot lines.
    spin_colors : Tuple[str], optional
        Colors for the spin texture lines.
    colorbar_title : str, optional
        Title of the colorbar.
    colorbar_title_size : int, optional
        Font size of the title of the colorbar.
    colorbar_title_padding : int, optional
        Padding of the title of the colorbar.
    colorbar_tick_labelsize : int, optional
        Size of the tick labels on the colorbar.

    Plot Appearance
    ---------------
    cmap : str, optional (default 'jet')
        The colormap used for the plot.
    clim : Tuple[float, float], optional
        The color scale limits for the color bar.
    fermi_color : str, optional
        Color of the Fermi line.
    fermi_linestyle : str, optional
        The linestyle of the Fermi line.
    fermi_linewidth : float, optional
        The linewidth of the Fermi line.
    grid : bool, optional
        If true, a grid will be shown on the plot.
    grid_axis : str, optional
        Which axis (or both) the grid lines should be drawn on.
    grid_color : str, optional
        The color of the grid lines.
    grid_linestyle : str, optional
        The linestyle of the grid lines.
    grid_linewidth : float, optional
        The linewidth of the grid lines.
    grid_which : str, optional
        Which grid lines to draw (major, minor, or both).
    label : Tuple[str], optional
        The labels for the plot lines.
    legend : bool, optional
        If true, a legend will be shown on the plot.
    linestyle : Tuple[str], optional
        The linestyles for the plot lines.
    linewidth : Tuple[float], optional
        The linewidths for the plot lines.
    marker : Tuple[str], optional
        The marker styles for the plot points.
    markersize : Tuple[float], optional
        The size of the markers for the plot points.
    opacity : Tuple[float], optional
        The opacities for the plot lines.
    plot_color_bar : bool, optional
        If true, a color bar will be shown on the plot.
    savefig : str, optional
        The file name to save the figure. If null, the figure will not be saved.
    title : str, optional
        The title for the plot. If null, no title will be displayed.
    weighted_color : bool, optional
        If true, the color of the lines will be weighted.
    weighted_width : bool, optional
        If true, the width of the lines will be weighted.
    figure_size : Tuple[int], optional
        The size of the figure (width, height) in inches.
    dpi : str, optional
        The resolution in dots per inch. If 'figure', use the figure's dpi value.
    
    
    colorbar_tick_params : Dict[str, any], optional
        The colorbar tick parameters, by default None
    colorbar_label_params : Dict[str, any], optional
        The colorbar label parameters, by default None
    x_label_params : Dict[str, any], optional
        The x label parameters, by default None
    y_label_params : Dict[str, any], optional
        The y label parameters, by default None
    title_params : Dict[str, any], optional
        The title parameters, by default None
    major_x_tick_params : Dict[str, any], optional
        The major x tick parameters, by default None
    major_y_tick_params : Dict[str, any], optional
        The major y tick parameters, by default None
    minor_y_tick_params : Dict[str, any], optional
        The minor y tick parameters, by default None
    major_y_locator : matplotlib.ticker.Locator, optional
        The major y locator, by default None
    minor_y_locator : matplotlib.ticker.Locator, optional
        The minor y locator, by default None
    multiple_locator_y_major_value : float, optional
        The major value for the multiple locator, by default None
    multiple_locator_y_minor_value : float, optional
        The minor value for the multiple locator, by default None
    

    Methods
    -------
    __post_init__():
        Post-initialization to set additional properties like `plot_type`.

    Examples
    --------
    To initialize a basic configuration with the default settings:

    >>> config = BandStructureConfig()

    To customize the plot with a specific colormap and line styles:

    >>> custom_config = BandStructureConfig(cmap='magma', linestyle=('dotted', 'dashed'))
    """
    modes: List[str] = field(default_factory=lambda: [mode.value for mode in BandStructureMode])
    # Basic Plot Settings
    color: str = 'black'
    spin_colors: Tuple[str] = field(default_factory=lambda: ('blue', 'red'))

    # Colorbar Configuration
    colorbar_title: str = 'Atomic Orbital Projections'
    colorbar_title_size: int = 15
    colorbar_title_padding: int = 20
    colorbar_tick_labelsize: int = 10

    # Plot Appearance
    cmap: str = 'jet'
    clim: Optional[Tuple[float, float]] = (0.0, 1.0)
    fermi_color: str = 'blue'
    fermi_linestyle: str = 'dotted'
    fermi_linewidth: float = 1
    grid: bool = False
    grid_axis: str = 'both'
    grid_color: str = 'grey'
    grid_linestyle: str = 'solid'
    grid_linewidth: float = 1
    grid_which: str = 'major'
    label: Tuple[str] = field(default_factory=lambda: (r'$\uparrow$', r'$\downarrow$'))
    legend: bool = True
    linestyle: Tuple[str] = field(default_factory=lambda: ('solid', 'dashed'))
    linewidth: Tuple[float] = field(default_factory=lambda: (1.0, 1.0))
    marker: Tuple[str] = field(default_factory=lambda: ('o', 'v', '^', 'D'))
    markersize: Tuple[float] = field(default_factory=lambda: (0.2, 0.2))
    opacity: Tuple[float] = field(default_factory=lambda: (1.0, 1.0))
    plot_color_bar: bool = True
    savefig: Optional[str] = None
    title: Optional[str] = None
    weighted_color: bool = True
    weighted_width: bool = False
    figure_size: Tuple[int] = field(default_factory=lambda: (9, 6))
    dpi: int = 300


    colorbar_tick_params: Dict[str, any] = field(default_factory=lambda: {})
    colorbar_label_params: Dict[str, any] = field(default_factory=lambda: {})
    
    x_label: str = 'K vector'
    x_label_params: Dict[str, any] = field(default_factory=lambda: {})
    y_label_params: Dict[str, any] = field(default_factory=lambda: {})
    title_params: Dict[str, any] = field(default_factory=lambda: {})
    # Tick Parameters
    major_y_tick_params: Dict[str, any] = field(default_factory=lambda: {
        "which": "major",
        "axis": "y",
        "direction": "inout",
        "width": 1,
        "length": 5,
        "labelright": False,
        "right": True,
        "left": True
    })
    minor_y_tick_params: Dict[str, any] = field(default_factory=lambda: {
        "which": "minor",
        "axis": "y",
        "direction": "in",
        "left": True,
        "right": True
    })
    major_x_tick_params: Dict[str, any] = field(default_factory=lambda: {
        "which": "major",
        "axis": "x",
        "direction": "in"
        })
    major_y_locator = None
    minor_y_locator = None

    multiple_locator_y_major_value:float = None
    multiple_locator_y_minor_value:float = None

    

    def __post_init__(self):
        """This method is immediately called after the object is initialized.
        It is useful to validate the data and set default values.
        """
        self.plot_type = PlotType.BAND_STRUCTURE

    def as_dict(self):
        """
        Returns a dictionary of the configuration settings.
        """
        return asdict(self)
