from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pyprocar.cfg.base import BaseConfig, PlotType


class BandStructure2DMode(Enum):
    """
    An enumeration for defining the modes of 2D band structure representations.

    Attributes
    ----------
    PLAIN : str
        Represents the band structure in a simple, where the colors are the different bands.
    PARAMETRIC : str
        Represents the band structure in a parametric form, summing over the projections.
    SPIN_TEXTURE : str
        Enhances the band structure representation with spin texture information

    """

    PLAIN = "plain"
    PARAMETRIC = "parametric"
    SPIN_TEXTURE = "spin_texture"


class BandStructure2DProperty(Enum):
    """
    An enumeration for defining the properties that can be visualized on the 2D band structure.

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
    AVG_INV_EFFECTIVE_MASS = "avg_inv_effective_mass"


@dataclass
class Bandstructure2DConfig(BaseConfig):
    """
    Configuration class for plotting 2D Band structures with customizable options for appearance and behavior.

    Plot Appearance
    ---------------
    background_color: str, optional (default 'white')
        Sets the background color of the plot.

    Surface Configuration
    ---------------------
    surface_cmap: str, optional (default 'jet')
        Defines the colormap for the surface plotting.
    surface_color: Optional[str], optional
        Specific color for the surface if not using a colormap.
    surface_spinpol_colors: List[str], optional
        Colors for each spin-polarized band, default is an empty list.
    surface_bands_colors: List[str], optional
        Colors for each band in the surface plot, default is an empty list.
    surface_opacity: float, optional (default 1.0)
        Sets the opacity level of the surface.
    surface_clim: Optional[List[float]], optional
        Defines the color limits for the surface colormap.

    Brillouin Zone and Super Cell
    -----------------------------
    brillouin_zone_style: str, optional (default 'wireframe')
        Defines the style of the Brillouin zone.
    brillouin_zone_line_width: float, optional (default 3.5)
        Sets the line width of the Brillouin zone lines.
    brillouin_zone_color: str, optional (default 'black')
        Specifies the color of the Brillouin zone lines.
    brillouin_zone_opacity: float, optional (default 1.0)
        Controls the opacity of the Brillouin zone lines.
    clip_brillouin_zone: bool, optional (default True)
        Whether to clip the Brillouin zone.
    clip_brillouin_zone_factor: float, optional (default 1.5)
        The factor to clip the Brillouin zone by.
    extended_zone_directions: Optional[List[List[int]]], optional
        List of directions to extend the surface in.
    supercell: List[int], optional (default [1, 1, 1])
        The supercell size to use.

    Texture and Axes
    ----------------
    texture_cmap: str, optional (default 'jet')
        Defines the colormap for texture mapping.
    texture_color: Optional[str], optional
        Specific color for the texture if not using a colormap.
    texture_size: float, optional (default 0.1)
        Size of the texture elements.
    texture_scale: bool, optional (default False)
        Flag to determine if texture scaling is applied.
    texture_opacity: float, optional (default 1.0)
        Opacity of the texture.
    show_axes: bool, optional (default True)
        Flag to determine if axes are added to the plot.
    x_axes_label: str, optional (default 'Kx')
        Label for the x-axis.
    y_axes_label: str, optional (default 'Ky')
        Label for the y-axis.
    z_axes_label: str, optional (default 'E')
        Label for the z-axis.

    Advanced Configurations
    -----------------------
    projection_accuracy: str, optional (default 'high')
        The accuracy of the projections. Options are 'high' and 'normal'.
    interpolation_factor: int, optional (default 1)
        The interpolation factor to use.

    Miscellaneous
    -------------
    plotter_offscreen: bool, optional (default False)
        Controls whether the plotter renders offscreen.
    plotter_camera_pos: List[int], optional (default [1, 1, 1])
        Specifies the camera position for the plotter.

    Axes Configuration
    -----------------
    axes_label_color: str, optional (default 'black')
        Color of the axes labels.
    axes_line_width: float, optional (default 6)
        Width of the axes lines.

    Grid Configuration
    -----------------
    show_grid: bool, optional (default True)
        Whether to show grid lines.
    grid_xtitle: str, optional (default 'k$_{x}$ ($\AA^{-1}$)')
        X-axis grid title.
    grid_ytitle: str, optional (default 'k$_{y}$ ($\AA^{-1}$)')
        Y-axis grid title.
    grid_ztitle: str, optional (default 'Energy (eV)')
        Z-axis grid title.
    grid_font_size: int, optional (default 10)
        Font size for grid labels.

    Fermi Plane Configuration
    ------------------------
    add_fermi_plane: bool, optional (default False)
        Whether to add a plane at the Fermi level.
    fermi_plane_opacity: float, optional (default 0.25)
        Opacity of the Fermi plane.
    fermi_plane_color: str, optional (default 'black')
        Color of the Fermi plane.
    fermi_plane_size: float, optional (default 0.5)
        Size of the Fermi plane.
    show_fermi_plane_text: bool, optional (default True)
        Whether to show text label for Fermi plane.
    fermi_text_position: List[float], optional (default [0,2,0])
        Position of the Fermi level text label.

    Scalar Bar Configuration
    -----------------------
    show_scalar_bar: bool, optional (default True)
        Whether to show the scalar bar (colorbar).
    scalar_bar_config: dict, optional (default {})
        Configuration for the scalar bar. See the PyVista documentation for
        `add_scalar_bar`(https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_scalar_bar#pyvista.Plotter.add_scalar_bar)`_
        for more information

    Animation Configuration
    ----------------------
    save_gif_config: dict, optional (default {})
        Configuration for the GIF animation. The arguments are
        `generate_orbital_path_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.generate_orbital_path)_` : dict
        `orbit_on_path_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.orbit_on_path)_` : dict
        `open_gif_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.open_gif)_` : dict

    save_mp4_config: dict, optional (default {})
        Configuration for the MP4 animation. The arguments are
        `generate_orbital_path_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.generate_orbital_path)_`
        `open_movie_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.open_movie)_`
        `orbit_on_path_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.orbit_on_path)_`

    save_mesh_config: dict, optional (default {})
        Configuration for the mesh saving. The arguments are
        `save_meshio_kwargs (https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.save_meshio)_`


    Cross Section Configuration
    -------------------------
    cross_section_slice_linewidth: float, optional (default 5.0)
        Line width for cross section slice.
    cross_section_slice_show_area: bool, optional (default False)
        Whether to show the cross section area.

    Methods
    -------
    as_dict():
        Returns a dictionary of the configuration settings.
    """

    modes: List[str] = field(
        default_factory=lambda: [mode.value for mode in BandStructure2DMode]
    )

    mode: BandStructure2DMode = BandStructure2DMode.PLAIN
    property: BandStructure2DProperty = BandStructure2DProperty.FERMI_SPEED
    # Basic Plot Settings
    background_color: str = "white"
    plotter_offscreen: bool = False
    plotter_camera_pos: List[int] = None

    # Surface Appearance
    surface_cmap: str = "jet"
    surface_color: Optional[str] = None
    surface_spinpol_colors: List[str] = field(default_factory=list)
    surface_bands_colors: List[str] = field(default_factory=list)
    surface_opacity: float = 1.0
    surface_clim: Optional[List[float]] = None

    # Brillouin Zone Styling
    show_brillouin_zone: bool = True
    brillouin_zone_style: str = "wireframe"
    brillouin_zone_line_width: float = 3.5
    brillouin_zone_color: str = "black"
    brillouin_zone_opacity: float = 1.0
    extended_zone_directions: Optional[List[List[int]]] = None
    supercell: List[int] = field(default_factory=lambda: [1, 1, 1])
    clip_brillouin_zone: bool = True
    clip_brillouin_zone_factor: float = 1.5

    # Texture and Axes
    texture_cmap: str = "jet"
    texture_color: Optional[str] = None
    texture_size: float = 0.1
    texture_scale: bool = False
    texture_clim: Optional[List[float]] = None
    texture_opacity: float = 1.0
    add_axes: bool = True
    x_axes_label: str = "Kx"
    y_axes_label: str = "Ky"
    z_axes_label: str = "E"
    energy_lim: List[int] = field(default_factory=lambda: [-2, 2])

    # Advanced Configurations
    projection_accuracy: str = "high"
    interpolation_factor: int = 1

    # Scalar Bar Configuration
    show_scalar_bar: bool = True
    scalar_bar_config: dict = field(
        default_factory=lambda: {
            "italic": False,
            "bold": False,
            "title_font_size": 15,
            "label_font_size": 10,
            "width": 0.5,
            "height": 0.05,
            "position_x": 0.48,
            "position_y": 0.02,
            "color": "black",
        }
    )

    # Axes Configuration
    show_axes: bool = True
    axes_label_color: str = "black"
    axes_line_width: float = 6.0

    # Grid Configuration
    show_grid: bool = True
    grid_xtitle: str = "k$_{x}$ ($\AA^{-1}$)"
    grid_ytitle: str = "k$_{y}$ ($\AA^{-1}$)"
    grid_ztitle: str = "Energy (eV)"
    grid_font_size: int = 10

    # Fermi Plane Configuration
    add_fermi_plane: bool = False
    fermi_plane_opacity: float = 0.25
    fermi_plane_color: str = "black"
    fermi_plane_size: float = 0.5
    show_fermi_plane_text: bool = True
    fermi_text_position: List[float] = field(default_factory=lambda: [0, 2, 0])

    # Animation Configuration
    save_gif_config: dict = field(
        default_factory=lambda: {
            "generate_orbital_path_kwargs": {"n_points": 36},
            "open_gif_kwargs": {},
            "orbit_on_path_kwargs": {"step": 0.05, "viewup": [0, 0, 1]},
        }
    )

    save_mp4_config: dict = field(
        default_factory=lambda: {
            "generate_orbital_path_kwargs": {"n_points": 36},
            "open_movie_kwargs": {},
            "orbit_on_path_kwargs": {"step": 0.05, "viewup": [0, 0, 1]},
        }
    )

    save_mesh_config: dict = field(
        default_factory=lambda: {
            "save_meshio_kwargs": {},
        }
    )

    # Isoslider Configuration
    isoslider_title: str = "Energy iso-value"
    isoslider_style: str = "modern"
    isoslider_color: str = "black"

    # Cross Section Configuration
    cross_section_slice_linewidth: float = 5.0
    cross_section_slice_show_area: bool = False

    def __post_init__(self):
        """This method is immediately called after the object is initialized.
        It is useful to validate the data and set default values.
        """
        self.plot_type = PlotType.BAND_STRUCTURE_2D

    def as_dict(self):
        """
        Returns a dictionary of the configuration settings.
        """
        return asdict(self)
