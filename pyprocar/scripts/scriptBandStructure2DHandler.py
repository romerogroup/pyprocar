__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import sys
from typing import cast

import numpy as np
import pyvista as pv

from pyprocar.cfg import ConfigManager
from pyprocar.cfg.band_structure_2d import Bandstructure2DConfig
from pyprocar.cfg.base import PlotType
from pyprocar.core import BandStructure2D
from pyprocar.plotter import BS2DPlotter
from pyprocar.utils import welcome
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


pv.global_theme.font.size = 10
np.set_printoptions(threshold=sys.maxsize)


class BandStructure2DHandler:
    """Handler for 2D band structure plotting.

    This class provides methods to create and visualize 2D band structure
    surfaces using the modernized BandStructure2D class.
    """

    def __init__(
        self,
        code: str,
        dirname: str = "",
        fermi: float | None = None,
        fermi_shift: float = 0,
        verbose: int = 1,
    ):
        """
        Initialize the BandStructure2D handler.

        Parameters
        ----------
        code : str
            The DFT code name ('vasp', 'qe', etc.)
        dirname : str, optional
            Directory containing the calculation files, by default ""
        fermi : float | None, optional
            The Fermi energy. If provided, bands will be shifted, by default None
        fermi_shift : float, optional
            Additional Fermi energy shift, by default 0
        verbose : int, optional
            Verbosity level, by default 1
        """
        set_verbose_level(verbose)

        user_logger.info("If you want more detailed logs, set verbose to 2 or more")
        user_logger.info("_" * 100)

        welcome()

        user_logger.info("_" * 100)

        self.default_config: Bandstructure2DConfig = Bandstructure2DConfig(plot_type=PlotType.BAND_STRUCTURE_2D)

        modes = ["plain", "parametric", "spin_texture"]
        props = ["bands_speed", "bands_velocity", "avg_inv_effective_mass"]
        modes_txt = " , ".join(modes)
        props_txt = " , ".join(props)
        self.notification_message: str = f"""
                There are additional plot options that are defined in a configuration file.
                You can change these configurations by passing the keyword argument to the function
                To print a list of plot options set print_plot_opts=True

                Here is a list modes : {modes_txt}
                Here is a list of properties: {props_txt}
                """

        self.code: str = code
        self.dirname: str = dirname
        self.fermi: float | None = fermi
        self.fermi_shift: float = fermi_shift

        # Set up energy labels based on Fermi level
        self.fermi_level: float | None
        self.energy_label: str
        self.fermi_message: str | None
        if fermi is not None:
            self.fermi_level = fermi_shift
            self.energy_label = r"E - E$_F$ (eV)"
            self.fermi_message = None
        else:
            self.energy_label = r"E (eV)"
            self.fermi_level = None
            self.fermi_message = (
                "`fermi` is not set! Set `fermi={value}`. "
                "The plot did not shift the bands by the Fermi energy."
            )

    def plot_band_structure(
        self,
        mode: str,
        bands: list[int] | None = None,
        _atoms: list[int] | None = None,
        _orbitals: list[int] | None = None,
        _spins: list[int] | None = None,
        spin_texture: bool = False,
        property_name: str | None = None,
        normal: tuple[float, float, float] = (0, 0, 1),
        origin: tuple[float, float, float] = (0, 0, 0),
        grid_interpolation: tuple[int, int] = (120, 120),
        show: bool = True,
        k_plane_scale: float = 2 * np.pi,
        render_offscreen: bool = False,
        save_2d: str | None = None,
        save_gif: str | None = None,
        save_mp4: str | None = None,
        save_3d: str | None = None,
        print_plot_opts: bool = False,
        **kwargs: object,
    ) -> None:
        """Plot 2D band structure surface.

        Parameters
        ----------
        mode : str
            The plotting mode ('plain', 'parametric', 'spin_texture')
        bands : list[int] | None, optional
            List of band indices to plot, by default None (uses bands near Fermi)
        _atoms : list[int] | None, optional
            List of atom indices for projections, by default None
        _orbitals : list[int] | None, optional
            List of orbital indices for projections, by default None
        _spins : list[int] | None, optional
            List of spin indices, by default None
        spin_texture : bool, optional
            Whether to plot spin texture, by default False
        property_name : str | None, optional
            Property to compute and display, by default None
        normal : tuple, optional
            Normal vector defining the cutting plane, by default (0, 0, 1)
        origin : tuple, optional
            Origin point of the cutting plane, by default (0, 0, 0)
        grid_interpolation : tuple, optional
            Number of grid points in (u, v) directions, by default (120, 120)
        show : bool, optional
            Whether to show the plot, by default True
        k_plane_scale : float, optional
            Scale factor for k-plane, by default 2π
        render_offscreen : bool, optional
            Whether to render offscreen, by default False
        save_2d : str | None, optional
            Path to save 2D screenshot, by default None
        save_gif : str | None, optional
            Path to save GIF animation, by default None
        save_mp4 : str | None, optional
            Path to save MP4 video, by default None
        save_3d : str | None, optional
            Path to save 3D mesh, by default None
        print_plot_opts : bool, optional
            Whether to print plotting options, by default False
        **kwargs
            Additional keyword arguments for configuration
        """
        config_base = ConfigManager.merge_configs(self.default_config, kwargs)
        config_base = ConfigManager.merge_config(config_base, "mode", mode)
        config = cast("Bandstructure2DConfig", config_base)

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        if self.fermi_message:
            user_logger.info(self.fermi_message)

        # Create BandStructure2D using modernized factory
        bs2d = BandStructure2D.from_code(
            code=self.code,
            dirpath=self.dirname,
            normal=normal,
            origin=origin,
            grid_interpolation=grid_interpolation,
            reduce_bands_near_fermi=True,
            bands=bands,
            scale_factor=k_plane_scale,
        )

        # Compute requested property
        if property_name is not None:
            bs2d.get_property(property_name)

        # Create plotter
        plotter = BS2DPlotter(bs2d, **kwargs)
        plotter.off_screen = render_offscreen

        # Add Brillouin zone if configured
        if config.show_brillouin_zone:
            bz = bs2d.get_2d_brillouin_zone(e_min=-2, e_max=2, scale_factor=k_plane_scale)
            plotter.add_brillouin_zone(bz)

        # Clip to Brillouin zone if configured
        if config.clip_brillouin_zone and plotter.brillouin_zone is not None:
            bs2d = plotter.clip_surface(bs2d, plotter.brillouin_zone)

        # Add surface to plotter
        plotter.add_surface(bs2d)

        # Add scalar bar if needed
        if (mode != "plain" or spin_texture) and config.show_scalar_bar:
            plotter.add_scalar_bar()

        # Add grid if configured
        if config.show_grid:
            plotter.show_grid(zlabel=self.energy_label)

        # Add axes if configured
        if config.show_axes:
            plotter.add_axes()

        # Add Fermi plane if available
        if self.fermi_level is not None:
            plotter.add_mesh(
                pv.Plane(
                    center=(0, 0, self.fermi_level),
                    direction=(0, 0, 1),
                    i_size=10,
                    j_size=10,
                ),
                color="red",
                opacity=0.5,
                name="fermi_plane",
            )

        plotter.set_background(color="white")

        # Handle output
        if save_2d:
            plotter.screenshot(filename=save_2d)
            plotter.close()
            return

        if show and not (save_gif or save_mp4 or save_3d):
            plotter.show()

        if save_gif:
            plotter.open_gif(save_gif)
            plotter.orbit_on_path(step=0.05)
            plotter.close()
        if save_mp4:
            plotter.open_movie(save_mp4)
            plotter.orbit_on_path(step=0.05)
            plotter.close()
        if save_3d:
            bs2d.save(save_3d)

        plotter.close()

    def print_default_settings(self) -> None:
        """
        Prints all the configuration settings with their current values.
        """
        for key, value in self.default_config.as_dict().items():
            user_logger.info(f"{key}: {value}")


# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx
