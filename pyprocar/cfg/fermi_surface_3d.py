from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pyprocar.cfg.base import BaseConfig, PlotType


class FermiSurfaceMode(Enum):
    """
    An enumeration for defining the modes of Fermi surface representations.

    Attributes
    ----------
    PLAIN : str
        Represents the Fermi surface in a simple, where the colors are the different bands.
    PARAMETRIC : str
        Represents the Fermi surface in a parametric form, summing over the projections.
    SPIN_TEXTURE : str
        Enhances the Fermi surface representation with spin texture information
    """

    PLAIN = "plain"
    PARAMETRIC = "parametric"
    SPIN_TEXTURE = "spin_texture"


class FermiSurfaceProperty(Enum):
    """
    An enumeration for defining the properties that can be visualized on the Fermi surface.

    Attributes
    ----------
    FERMI_SPEED : str
        Visualizes the scalar speed at which electrons travel at the Fermi level.
    FERMI_VELOCITY : str
        Visualizes the vector velocity of electrons moving at the Fermi level.
    AVG_INV_EFFECTIVE_MASS : str
        Displays the average inverse effective mass of electrons, reflecting the electron mobility.
    """

    FERMI_SPEED = "fermi_speed"
    FERMI_VELOCITY = "fermi_velocity"
    AVG_INV_EFFECTIVE_MASS = "avg_inv_effective_mass"


@dataclass
class FermiSurface3DConfig(BaseConfig):
    """
    Configuration class for plotting 3D Fermi surfaces in different modes and styles.

    Parameters
    ----------
    mode : FermiSurfaceMode, optional (default `FermiSurfaceMode.PLAIN`)
        Defines the mode of the Fermi surface representation. Options are:
        `PLAIN`, `PARAMETRIC`, and `SPIN_TEXTURE`.
    property : FermiSurfaceProperty, optional (default `FermiSurfaceProperty.FERMI_SPEED`)

    Plot Appearance
    ---------------
    background_color : str, optional (default 'white')
        Sets the background color of the plot.
    plotter_offscreen : bool, optional (default False)
        Determines whether the plotter renders offscreen.
    plotter_camera_pos : List[int], optional (default [1, 1, 1])
        Specifies the camera position for the plotter.

    Surface Configuration
    ---------------------
    surface_cmap : str, optional (default 'jet')
        Defines the colormap for the surface plotting.
    surface_color : str, optional (default None)
        Specific color for the surface if not using a colormap.
    surface_opacity : float, optional (default 1.0)
        Sets the opacity level of the surface.
    surface_clim : List[float], optional (default None)
        Defines the color limits for the surface colormap.
    surface_bands_colors : List[str], optional (default [])
        List of colors for each band in the surface plot.



    Spin Settings
    ---------------------
    spin_colors : Tuple[str], optional (None, None)
        List of colors for the spin texture lines.
    texture_cmap : str, optional
        Defines the colormap for texture mapping, default is "jet".
    texture_color : Optional[str], optional
        Specific color for the texture if not using a colormap, default is None.
    texture_size : float, optional
        Size of the texture elements, default is 0.1.
    texture_scale : bool, optional
        Flag to determine if texture scaling is applied, default is False.
    texture_opacity : float, optional
        Opacity of the texture, default is 1.0.
    arrow_size : int, optional (default 3)
        Size of the arrows in the spin texture mode.


    Axes and Labels
    ---------------
    add_axes : bool, optional
        Flag to determine if axes are added to the plot, default is True.
    x_axes_label : str, optional
        Label for the x-axis, default is "Kx".
    y_axes_label : str, optional
        Label for the y-axis, default is "Ky".
    z_axes_label : str, optional
        Label for the z-axis, default is "Kz".
    axes_label_color : str, optional
        Color of the axes labels, default is "black".
    axes_line_width : float, optional
        Line width of the axes, default is 6.


    Brillouin Zone Styling
    ----------------------
    brillouin_zone_style : str, optional (default "wireframe")
        Defines the style of the Brillouin zone.
    brillouin_zone_line_width : float, optional (default 3.5)
        Sets the line width of the Brillouin zone lines.
    brillouin_zone_color : str, optional (default "black")
        Specifies the color of the Brillouin zone lines.
    brillouin_zone_opacity : float, optional (default 1.0)
        Controls the opacity of the Brillouin zone lines.

    Advanced Configurations
    -----------------------
    fermi_tolerance : float, optional
        The tolerance to search for bands around the Fermi energy, default is 0.1.
    extended_zone_directions : Optional[List[List[int]]], optional
        List of directions to extend the surface in, default is None.
    supercell : List[int], optional
        The supercell size to use for the Fermi surface, default is [1, 1, 1].
    projection_accuracy : str, optional
        The accuracy of the projections with options 'high' and 'normal', default is 'high'.
    interpolation_factor : int, optional
        The interpolation factor to use for the Fermi surface, default is 1.

    Advanced Configurations
    -----------------------
    fermi_tolerance : float, optional (default 0.1)
        The tolerance to search for bands around the fermi energy.
    extended_zone_directions : List[List[int]], optional (default None)
        List of directions to extend the surface in.
    supercell : List[int], optional (default [1, 1, 1])
        The supercell size to use for the Fermi surface.
    projection_accuracy : str, optional (default 'high')
        The accuracy of the projections. Options are 'high' and 'normal'.
    interpolation_factor : int, optional (default 1)
        The interpolation factor to use for the Fermi surface.
    max_distance : float, optional (default 0.2)
        The maximum distance to keep points from the isosurface centers.

    Cross section Settings
    ----------------------
    cross_section_slice_linewidth : float, optional (default 5.0)
        Line width for slices in cross-sectional views.
    cross_section_slice_show_area : bool, optional (default False)
        Flag to determine if the area under the cross-section should be shown.

    Isoslider Settings
    -------------------
    isoslider_title : str, optional (default 'Energy iso-value')
        Title for the iso-value slider in the interface.
    isoslider_style : str, optional (default 'modern')
        Style of the iso-value slider.
    isoslider_color : str, optional (default 'black')
        Color of the iso-value slider.

    Miscellaneous
    -------------
    orbit_gif_n_points : int, optional
        Number of points to interpolate for creating orbit GIF animations, default is 36.
    orbit_gif_step : float, optional
        Step size between points in the orbit GIF animation, default is 0.05.
    orbit_mp4_n_points : int, optional
        Number of points to interpolate for creating orbit MP4 animations, default is 36.
    orbit_mp4_step : float, optional
        Step size between points in the orbit MP4 animation, default is 0.05.

    Methods
    -------
    __post_init__():
        Post-initialization to set additional properties like `plot_type`.

    Examples
    --------
    To initialize a basic configuration with the default settings:

    >>> config = FermiSurface3DConfig()

    To customize the Fermi surface with a specific colormap and opacity:

    >>> custom_config = FermiSurface3DConfig(surface_cmap='magma', surface_opacity=0.8)
    """

    # Basic Plot Settings
    mode: FermiSurfaceMode = FermiSurfaceMode.PLAIN
    property: FermiSurfaceProperty = None
    property_name: str = None
    background_color: str = "white"
    plotter_offscreen: bool = False
    plotter_camera_pos: List[int] = field(default_factory=lambda: [1, 1, 1])

    # Surface Appearance
    surface_cmap: str = "jet"  # Colormap for the surface
    surface_color: Optional[str] = None  # Specific color for the surface
    surface_opacity: float = 1.0
    surface_clim: Optional[List[float]] = None
    surface_bands_colors: List[str] = field(default_factory=list)

    # Spin Settings
    spin_colors: Optional[Tuple[str]] = (None, None)
    arrow_size: int = 3  # Size of arrows for spin texture
    texture_cmap: str = "jet"
    texture_color: Optional[str] = None
    texture_size: float = 0.05
    texture_scale: bool = False
    texture_clim: Optional[List[float]] = None
    texture_opacity: float = 1.0

    # Brillouin Zone Styling
    show_brillouin_zone: bool = True
    brillouin_zone_style: str = "wireframe"
    brillouin_zone_line_width: float = 3.5
    brillouin_zone_color: str = "black"
    brillouin_zone_opacity: float = 1.0

    # Axes and Labels
    show_axes: bool = True
    x_axes_label: str = "Kx"
    y_axes_label: str = "Ky"
    z_axes_label: str = "Kz"
    axes_label_color: str = "black"
    axes_line_width: float = 6

    # Scalar Bar Configurations
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

    # Advanced Configurations
    fermi_tolerance: float = 0.1
    extended_zone_directions: Optional[List[List[int]]] = None
    supercell: List[int] = field(default_factory=lambda: [1, 1, 1])
    projection_accuracy: str = "high"
    interpolation_factor: int = 1
    max_distance: float = 0.3

    # Cross section Settings
    cross_section_slice_linewidth: float = 5.0
    cross_section_slice_show_area: bool = False

    # Isoslider Settings
    isoslider_title: str = "Energy iso-value"
    isoslider_style: str = "modern"
    isoslider_color: str = "black"

    # Miscellaneous
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

    def __post_init__(self):
        """This method is immediately called after the object is initialized.
        It is useful to validate the data and set default values.
        """
        self.plot_type = PlotType.FERMI_SURFACE_3D

    def as_dict(self):
        """
        Returns a dictionary of the configuration settings.
        """
        return asdict(self)
