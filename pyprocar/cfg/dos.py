from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pyprocar.cfg.base import BaseConfig, PlotType


class DOSMode(Enum):
    """
    An enumeration for defining the modes of density of states representations.

    Attributes
    ----------
    PLAIN : str
        Represents the density of states in a simple, where the colors are the different bands.
    PARAMETRIC : str
        Represents the density of states in a parametric form, summing over the projections.
    PARAMETRIC_LINE : str
        Represents the density of states in a parametric line form, summing over the projections.
    STACK : str
        Represents the density of states in a stack plot, where the colors projection contributions to the DOS.
    STACK_ORBITALS : str
        Represents the density of states in a stack plot, where the colors are the different orbitals contributions to the DOS.
    STACK_SPECIES : str
        Represents the density of states in a stack plot, where the colors are the different species contributions to the DOS.
    OVERLAY : str
        Represents the density of states in an overlay plot, where the colors are the selected projections
    OVERLAY_ORBITALS : str
        Represents the density of states in an overlay plot, where the colors are
        the different projection of the orbitals.
    OVERLAY_SPECIES : str
        Represents the density of states in an overlay plot, where the colors are
        the different projection of the species.
    """

    PLAIN = "plain"
    PARAMETRIC = "parametric"
    PARAMETRIC_LINE = "parameteric_line"
    STACK = "stack"
    STACK_ORBITALS = "stack_orbitals"
    STACK_SPECIES = "stack_species"
    OVERLAY = "overlay"
    OVERLAY_ORBITALS = "overlay_orbitals"
    OVERLAY_SPECIES = "overlay_species"


@dataclass
class DensityOfStatesConfig(BaseConfig):
    """
    Configuration class for plotting the density of states with various options.

    Parameters
    ----------
    cmap : str, optional (default 'jet')
        The colormap used for the plot.
    colors : List[str], optional
        List of colors for the plot lines.
    color : List[str], optional
        List of colors for the spin up and spin down lines.
    colorbar_title : str, optional
        Title of the colorbar.
    colorbar_title_size : int, optional
        Font size of the title of the colorbar.
    colorbar_title_padding : int, optional
        Padding of the title of the colorbar.
    colorbar_tick_labelsize : int, optional
        Size of the tick labels on the colorbar.
    fermi_color : str, optional
        Color of the Fermi line.
    fermi_linestyle : str, optional
        The linestyle of the Fermi line.
    fermi_linewidth : float, optional
        The linewidth of the Fermi line.
    figure_size : Tuple[int, int], optional
        The size of the figure (width, height) in inches.
    font : str, optional
        The font style for the plot text.
    font_size : int, optional
        The size of the font used in the plot.
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
    legend : bool, optional
        If true, a legend will be shown on the plot.
    linestyle : List[str], optional
        The linestyles for the plot lines.
    linewidth : List[float], optional
        The linewidths for the plot lines.
    marker : List[str], optional
        The marker styles for the plot points.
    markersize : List[float], optional
        The size of the markers for the plot points.
    opacity : List[float], optional
        The opacities for the plot lines.
    plot_bar : bool, optional
        If true, a bar plot will be displayed.
    plot_color_bar : bool, optional
        If true, a color bar will be shown on the plot.
    plot_total : bool, optional
        If true, the total plot will be displayed.
    savefig : str, optional
        The file name to save the figure. If null, the figure will not be saved.
    spin_colors : List[str], optional
        The colors for the spin up and spin down lines.
    spin_labels : List[str], optional
        The labels for the spin up and spin down.
    title : str, optional
        The title for the plot. If null, no title will be displayed.
    verbose : bool, optional
        If true, the program will print detailed information.
    weighted_color : bool, optional
        If true, the color of the lines will be weighted.
    weighted_width : bool, optional
        If true, the width of the lines will be weighted.
    clim : Tuple[float, float], optional
        Value range to scale the colorbar.
    stack_y_label : str, optional
        The label for the y-axis for stack mode.
    x_label : str, optional
        The label for the x-axis.
    y_label : str, optional
        The label for the y-axis.
    dpi : str, optional
        The resolution in dots per inch. If 'figure', use the figure's dpi value.

    Methods
    -------
    __post_init__():
        Post-initialization to set additional properties like `plot_type`.

    Examples
    --------
    To initialize a basic configuration with the default settings:

    >>> config = DensityOfStatesConfig()

    To customize the plot with specific colors and Fermi line settings:

    >>> custom_config = DensityOfStatesConfig(colors=['red', 'blue'], fermi_color='black', fermi_linestyle='dotted')
    """

    modes: List[str] = field(default_factory=lambda: [mode.value for mode in DOSMode])
    # Basic Plot Settings
    cmap: str = "jet"
    colors: List[str] = field(
        default_factory=lambda: [
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
            "orange",
            "purple",
            "brown",
            "navy",
            "maroon",
            "olive",
        ]
    )

    color: str = "black"
    colorbar_title: str = "Atomic Orbital Projections"
    colorbar_title_size: int = 15
    colorbar_title_padding: int = 20
    colorbar_tick_labelsize: int = 10
    fermi_color: str = "black"
    fermi_linestyle: str = "dotted"
    fermi_linewidth: float = 1
    figure_size: Tuple[int, int] = (9, 6)
    font: str = "Arial"
    font_size: int = 16
    grid: bool = False
    grid_axis: str = "both"
    grid_color: str = "grey"
    grid_linestyle: str = "solid"
    grid_linewidth: float = 1
    grid_which: str = "major"
    draw_baseline: bool = True
    baseline_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "color": "black",
            "alpha": 0.3,
            "linestyle": "solid",
            "linewidth": 1,
        }
    )

    legend: bool = True
    linestyle: List[str] = field(default_factory=lambda: ["solid", "dashed"])
    linewidth: List[float] = field(default_factory=lambda: [1, 1])
    marker: List[str] = field(default_factory=lambda: ["o", "v", "^", "D"])
    markersize: List[float] = field(default_factory=lambda: [0.2, 0.2])
    opacity: List[float] = field(default_factory=lambda: [1.0, 1.0])
    plot_bar: bool = True
    plot_color_bar: bool = True
    plot_total: bool = True
    savefig: Optional[str] = None
    spin_colors: List[str] = field(default_factory=lambda: ["black", "red"])
    spin_labels: List[str] = field(
        default_factory=lambda: [r"$\uparrow$", r"$\downarrow$"]
    )
    title: Optional[str] = None
    verbose: bool = True
    weighted_color: bool = True
    weighted_width: bool = False
    clim: Optional[Tuple[float, float]] = None
    stack_y_label: str = "DOS"
    x_label: str = ""
    y_label: str = ""
    dpi: int = 300

    x_label_params: Dict[str, Any] = field(default_factory=lambda: {})
    y_label_params: Dict[str, Any] = field(default_factory=lambda: {})
    legend_params: Dict[str, Any] = field(default_factory=lambda: {})

    major_x_tick_params: Dict[str, Any] = field(default_factory=lambda: {})
    minor_x_tick_params: Dict[str, Any] = field(default_factory=lambda: {})
    major_y_tick_params: Dict[str, Any] = field(default_factory=lambda: {})
    minor_y_tick_params: Dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        """This method is immediately called after the object is initialized.
        It is useful to validate the data and set default values.
        """
        self.plot_type = PlotType.DENSITY_OF_STATES

    def as_dict(self):
        """
        Returns a dictionary of the configuration settings.
        """
        return asdict(self)
