from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.base import BaseConfig, PlotType


class UnfoldPlotMode(Enum):
    """
    An enumeration for defining the modes of unfolding representations in reciprocal space.

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


class UnfoldMode(Enum):
    """
    An enumeration for defining the properties that can be visualized on the BandStructure.

    Attributes
    ----------
    BOTH : str
        Represents the band structure in a simple, where the colors are the different bands.
    THICKNESS : str
        Represents the band structure in a simple, where the colors are the different bands.
    COLOR : str
        Represents the band structure in a simple, where the colors are the different bands.
    """

    BOTH = "both"
    THICKNESS = "thickness"
    COLOR = "color"


@dataclass
class UnfoldingConfig(BandStructureConfig):
    r"""
    Configuration class for plotting unfolding diagrams with various visual properties.

    Parameters
    ----------
    color: str, optional (default 'black')
        Sets the overall color for plot lines.

    Plot Appearance
    ---------------
    spin_colors: Tuple[str], optional
        Tuple of colors for spin-dependent plot lines, default is ('blue', 'red').
    fermi_color: str, optional (default 'blue')
        Color of the Fermi line on the plot.
    grid_color: str, optional (default 'grey')
        Color of the grid lines on the plot.

    Line Styles
    -----------
    fermi_linestyle: str, optional (default 'dotted')
        Linestyle for the Fermi line.
    grid_linestyle: str, optional (default 'solid')
        Linestyle for the grid lines.
    linestyle: List[str], optional
        List of linestyles for plot lines, default is ['solid', 'dashed'].

    Line Widths
    -----------
    fermi_linewidth: int, optional (default 1)
        Linewidth for the Fermi line.
    grid_linewidth: int, optional (default 1)
        Linewidth for the grid lines.
    linewidth: List[float], optional
        List of linewidths for plot lines, default is [1.0, 1.0].

    Markers
    -------
    marker: List[str], optional
        List of marker styles for plot points, default is ['o', 'v', '^', 'D'].
    markersize: List[float], optional
        List of sizes for the markers, default is [0.2, 0.2].

    Color and Opacity Settings
    --------------------------
    cmap: str, optional (default 'jet')
        Colormap used for the plot.
    clim: Tuple[float], optional
        Color scale limits for the color bar.
    opacity: List[float], optional
        Opacities for the plot lines, default is [1.0, 1.0].
    plot_color_bar: bool, optional (default True)
        If true, a color bar will be shown on the plot.

    Grid and Legend
    ---------------
    grid: bool, optional (default False)
        If true, a grid will be shown on the plot.
    grid_axis: str, optional (default 'both')
        Specifies which axis the grid lines should be drawn on.
    grid_which: str, optional (default 'major')
        Specifies which grid lines to draw (major, minor, or both).
    legend: bool, optional (default True)
        If true, a legend will be displayed on the plot.

    Labels and Title
    ----------------
    label: List[str], optional
        Labels for the plot lines, default is [r'$\uparrow$', r'$\downarrow$'].
    title: Optional[str], optional
        Title for the plot.

    Miscellaneous
    -------------
    figure_size: Tuple[int], optional (default (9, 6))
        Size of the figure in inches.
    dpi: str, optional (default 'figure')
        The resolution in dots per inch.
    savefig: Optional[str], optional
        The file name to save the figure. If null, the figure will not be saved.

    Advanced Configurations
    -----------------------
    weighted_color: bool, optional (default True)
        If true, the color of the lines will be weighted.
    weighted_width: bool, optional (default False)
        If true, the width of the lines will be weighted.

    Methods
    -------
    __post_init__():
        Post-initialization to validate the data and set default values.
    """

    modes: List[str] = field(
        default_factory=lambda: [mode.value for mode in UnfoldMode]
    )
    # Basic Plot Settings
    color: str = "#eeeeee"
    spin_colors: Tuple[str] = ("blue", "red")
    fermi_color: str = "blue"
    grid_color: str = "grey"

    # Line Styles
    fermi_linestyle: str = "dotted"
    grid_linestyle: str = "solid"
    linestyle: List[str] = field(default_factory=lambda: ["solid", "dashed"])

    # Line Widths
    fermi_linewidth: int = 1
    grid_linewidth: int = 1
    linewidth: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Markers
    marker: List[str] = field(default_factory=lambda: ["o", "v", "^", "D"])
    markersize: List[float] = field(default_factory=lambda: [0.2, 0.2])

    # Color and Opacity Settings
    cmap: str = "jet"
    clim: Optional[Tuple[float, float]] = (0.0, 1.0)
    opacity: List[float] = field(default_factory=lambda: [0.3, 0.3])
    plot_color_bar: bool = True

    # Grid and Legend
    grid: bool = False
    grid_axis: str = "both"
    grid_which: str = "major"
    legend: bool = True

    # Labels and Title
    label: List[str] = field(default_factory=lambda: [r"$\uparrow$", r"$\downarrow$"])
    title: Optional[str] = None

    # Miscellaneous
    figure_size: Tuple[int] = (9, 6)
    dpi: str = "figure"
    savefig: Optional[str] = None

    # Advanced Configurations
    weighted_color: bool = True
    weighted_width: bool = False

    # label params
    x_label_params: Dict[str, any] = field(default_factory=lambda: {})
    y_label_params: Dict[str, any] = field(default_factory=lambda: {})
    title_params: Dict[str, any] = field(default_factory=lambda: {})

    # x tick parameters
    major_x_tick_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "direction": "in",
            "length": 4,
            "width": 1,
            "colors": "black",
        }
    )

    # y tick parameters
    major_y_tick_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "direction": "in",
            "length": 4,
            "width": 1,
            "colors": "black",
        }
    )
    minor_y_tick_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "direction": "in",
            "length": 2,
            "width": 1,
            "colors": "black",
        }
    )

    # locators
    major_y_locator = None
    minor_y_locator = None
    multiple_locator_y_major_value: float = None
    multiple_locator_y_minor_value: float = None

    weighted_width: bool = False
    weighted_color: bool = True

    def __post_init__(self):
        """Post-initialization to validate the data and set default values."""
        self.plot_type = PlotType.UNFOLD

    def as_dict(self):
        """Returns a dictionary of the configuration settings."""
        return asdict(self)
