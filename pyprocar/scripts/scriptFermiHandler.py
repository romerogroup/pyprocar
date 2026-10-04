__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import sys
from typing import cast

import numpy as np

from pyprocar.cfg import ConfigFactory, ConfigManager
from pyprocar.cfg.base import PlotType
from pyprocar.cfg.fermi_surface_3d import FermiSurface3DConfig
from pyprocar.core import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.plotter import FermiPlotter
from pyprocar.scripts._selection import as_clim, resolve_spins
from pyprocar.utils import welcome
from pyprocar.utils.log_utils import set_verbose_level, warn_user

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def _new_plotter(kwargs, save_2d) -> FermiPlotter:
    """Build the FermiPlotter, off screen when a screenshot is requested."""
    plotter_kwargs = {
        k: v for k, v in kwargs.items() if k in ["off_screen", "window_size", "theme"]
    }
    if save_2d:
        plotter_kwargs["off_screen"] = True
    return FermiPlotter(**plotter_kwargs)


class FermiHandler:
    def __init__(
        self,
        code: str,
        dirname: str = "",
        fermi: float | None = None,
        *,
        use_cache: bool = False,
        ebs_filename: str = "ebs.pkl",
        verbose: int = 1,
    ):
        """
        This class handles the plotting of the fermi surface. Initialize by specifying the code and directory name where the data is stored.
        Then call one of the plotting methods provided.

        Parameters
        ----------
        code : str
            The code name
        dirname : str, optional
            the directory name where the calculation is, by default ""
        fermi : float, optional
            The fermi energy. This will overide the default fermi value used found in the given directory, by default None
        use_cache : bool, optional
            Boolean to use cached Pickle files, by default False
        ebs_filename : str, optional
            Name of the cached band structure file, by default "ebs.pkl"
        verbose : int, optional
            Verbosity level, by default 1
        """

        set_verbose_level(verbose)
        welcome()

        user_logger.info("_" * 100)

        self.default_config = ConfigFactory.create_config(PlotType.FERMI_SURFACE_3D)

        self.code = code
        self.dirname = dirname
        self.ebs: ElectronicBandStructureMesh = cast(
            ElectronicBandStructureMesh,
            ElectronicBandStructureMesh.from_code(
                code, dirname, use_cache=use_cache, ebs_filename=ebs_filename
            ),
        )

        if fermi is None:
            self.e_fermi: float = self.ebs.fermi
            user_logger.info(
                f"Fermi Energy not set! Set `fermi={self.e_fermi}`."
                "By default, using fermi energy found in the current directory."
            )
        else:
            self.e_fermi = fermi

        modes = ["plain", "parametric", "spin_texture"]
        props = ["fermi_speed", "fermi_velocity", "avg_inv_effective_mass"]
        modes_txt = " , ".join(modes)
        props_txt = " , ".join(props)
        self.notification_message = f"""
                There are additional plot options that are defined in a configuration file. 
                You can change these configurations by passing the keyword argument to the function
                To print a list of plot options set print_plot_opts=True

                Here is a list modes : {modes_txt}
                Here is a list of properties: {props_txt}"""

    def _map_mode_to_property(
        self, mode, bands=None, atoms=None, orbitals=None, spins=None, spin_texture=False
    ):
        """
        Maps old mode system to new property system

        Parameters
        ----------
        mode : str
            The mode to map
        bands : List[int], optional
            A list of band indexes to plot, by default None
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False

        Returns
        -------
        str or None
            Property name to compute, or None for plain mode
        """
        if mode == "plain":
            return "spin_band_index"
        elif mode == "parametric":
            # For parametric mode, we need to compute projections
            return "projected_sum"
        elif mode == "spin_texture":
            if spin_texture:
                return "projected_sum_spin_texture"
            else:
                return "projected_sum"
        elif mode == "fermi_speed" or mode == "fermi_velocity":
            return mode
        elif mode == "overlay":
            return "projected_sum"
        else:
            warn_user(f"Unknown mode: {mode}. Using plain mode.")
            return None

    def _create_fermi_surface(
        self, fermi=None, fermi_shift=0.0, bands=None, atoms=None, orbitals=None, spins=None
    ):
        """
        Creates a FermiSurface object with the stored parameters

        Parameters
        ----------
        fermi : float, optional
            Fermi energy to use, by default None (uses self.e_fermi)
        fermi_shift : float, optional
            Energy shift to apply, by default 0.0
        bands : List[int], optional
            Bands to reduce to, by default None
        atoms : List[int], optional
            Atoms for projections, by default None
        orbitals : List[int], optional
            Orbitals for projections, by default None
        spins : List[int], optional
            Spins for projections, by default None

        Returns
        -------
        FermiSurface
            The created FermiSurface object
        """
        if fermi is None:
            fermi = self.e_fermi
        ebs = self.ebs if bands is None else self.ebs.reduce_bands_by_index(bands, inplace=False)
        return FermiSurface.from_ebs(ebs, isovalue=fermi, isovalue_shift=fermi_shift)

    def plot_fermi_surface(
        self,
        mode,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
        fermi_shift=0.0,
        show=True,
        save_2d=None,
        save_gif=None,
        save_mp4=None,
        save_3d=None,
        print_plot_opts: bool = False,
        show_colorbar: bool = False,
        **kwargs,
    ):
        """A method to plot the 3d fermi surface

        Parameters
        ----------
        mode : str
            The mode to calculate
        bands : List[int], optional
            A list of band indexes to plot, by default None
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        fermi_shift : float, optional
            Energy shift to apply to the Fermi level, by default 0.0
        show : bool, optional
            Whether to show the plot, by default True
        save_2d : str, optional
            Filename to save 2D plot, by default None
        save_gif : str, optional
            Filename to save GIF, by default None
        save_mp4 : str, optional
            Filename to save MP4, by default None
        save_3d : str, optional
            Filename to save 3D mesh, by default None
        print_plot_opts: bool, optional
            Boolean to print the plotting options
        """
        config = ConfigManager.merge_configs(self.default_config, kwargs)
        config = cast(FermiSurface3DConfig, ConfigManager.merge_config(config, "mode", mode))

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Create FermiSurface object
        fermi_surface = self._create_fermi_surface(
            fermi_shift=fermi_shift, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins
        )

        if fermi_surface.n_points == 0:
            warn_user(
                "No Fermi surface found for the given parameters. Skipping plotting."
            )
            return None

        property_name = self._map_mode_to_property(
            mode, bands, atoms, orbitals, spins, spin_texture
        )
        channels, projection_spins, _ = resolve_spins(
            self.ebs.is_non_collinear, self.ebs.n_spin_channels, spins, plain=mode == "plain"
        )
        scalars_data = vectors_data = None
        if mode == "plain":
            scalars_data = fermi_surface.get_property("spin_band_index")
        elif property_name:
            prop = fermi_surface.get_property(
                property_name, atoms=atoms, orbitals=orbitals, spins=projection_spins
            )
            if prop.value.shape[-1] == 3:
                vectors_data = prop
            else:
                scalars_data = prop

        fsplt = _new_plotter(kwargs, save_2d)
        fsplt.plot(
            fermi_surface,
            scalars_data=scalars_data,
            vectors_data=vectors_data,
            spins=channels,
            show_brillouin_zone=config.show_brillouin_zone,
            show_scalar_bar=(scalars_data is not None and mode != "plain") or show_colorbar,
            scalars_cmap=config.surface_cmap,
            scalars_clim=None if mode == "plain" else as_clim(config.surface_clim),
            add_surface_kwargs={"opacity": config.surface_opacity},
        )

        if config.show_axes:
            fsplt.add_axes(
                xlabel=config.x_axes_label,
                ylabel=config.y_axes_label,
                zlabel=config.z_axes_label,
            )

        # Handle saving and showing
        if save_2d:
            fsplt.savefig(filename=save_2d)
            return fsplt

        if show and (save_gif is None and save_mp4 is None and save_3d is None):
            fsplt.show()

        if save_gif is not None:
            warn_user("GIF saving not yet implemented in new API")

        if save_mp4:
            warn_user("MP4 saving not yet implemented in new API")

        if save_3d:
            fsplt.export_data(str(save_3d))
        return fsplt

    def plot_fermi_isoslider(
        self,
        mode,
        iso_range: float = None,
        iso_surfaces: int = None,
        iso_values: list[float] = None,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
        show=True,
        save_2d=None,
        print_plot_opts: bool = False,
        **kwargs,
    ):
        """A method to plot the 3d fermi surface with an energy slider

        Parameters
        ----------
        iso_range : float
            A range of energies the slide will go through
        iso_surfaces : int
            The number of fermi surfaces to calculate on the range
        iso_values : List[float], optional
            A list of energies the slider will go through
        mode : str
            The mode to calculate
        bands : List[int], optional
            A list of band indexes to plot, by default None
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        show : bool, optional
            Whether to show the plot, by default True
        save_2d : str, optional
            Filename to save 2D plot, by default None
        print_plot_opts: bool, optional
            Boolean to print the plotting options
        """
        config = ConfigManager.merge_configs(self.default_config, kwargs)
        config = ConfigManager.merge_config(config, "mode", mode)

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Determine energy values for the slider
        if iso_surfaces is not None and iso_range is not None:
            energy_values = np.linspace(
                self.e_fermi - iso_range / 2, self.e_fermi + iso_range / 2, iso_surfaces
            )
        elif iso_values:
            energy_values = iso_values
        else:
            raise ValueError("Either iso_surfaces and iso_range or iso_values must be provided")

        # Determine property to compute based on mode
        property_name = self._map_mode_to_property(
            mode, bands, atoms, orbitals, spins, spin_texture
        )

        # Create FermiSurface objects for each energy value
        fermi_surfaces = []
        for e_value in energy_values:
            fs = self._create_fermi_surface(
                fermi=e_value, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins
            )

            # Compute property if needed
            if property_name:
                prop = fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)
                fs.set_values(property_name, prop.value)

            fermi_surfaces.append(fs)

            logger.debug(f"___Generated surface for energy {e_value}___")
            logger.debug(f"Surface has {fs.n_points} points")

        # Create plotter and add isoslider
        fsplt = _new_plotter(kwargs, save_2d)

        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_isoslider(
            fermi_surfaces,
            energy_values,
            add_active_vectors=add_active_vectors,
            add_surface_args={
                "show_scalar_bar": config.show_scalar_bar and property_name is not None,
                "cmap": config.surface_cmap,
                "clim": config.surface_clim,
                "opacity": config.surface_opacity,
            },
        )

        # Handle saving and showing
        if save_2d:
            fsplt.savefig(filename=save_2d)
            return None

        if show:
            fsplt.show()

    def create_isovalue_gif(
        self,
        mode,
        iso_range: float = None,
        iso_surfaces: int = None,
        iso_values: list[float] = None,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
        save_gif=None,
        print_plot_opts: bool = False,
        **kwargs,
    ):
        """A method to create a GIF of fermi surfaces at different energies

        Parameters
        ----------
        iso_range : float
            A range of energies the GIF will go through
        iso_surfaces : int
            The number of fermi surfaces to calculate on the range
        iso_values : List[float], optional
            A list of energies the GIF will go through
        mode : str
            The mode to calculate
        bands : List[int], optional
            A list of band indexes to plot, by default None
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        save_gif : str, optional
            Filename to save GIF, by default None
        print_plot_opts: bool, optional
            Boolean to print the plotting options
        """
        config = ConfigManager.merge_configs(self.default_config, kwargs)
        config = ConfigManager.merge_config(config, "mode", mode)

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Determine energy values for the GIF
        if iso_surfaces is not None and iso_range is not None:
            energy_values = np.linspace(
                self.e_fermi - iso_range / 2, self.e_fermi + iso_range / 2, iso_surfaces
            )
        elif iso_values:
            energy_values = iso_values
        else:
            raise ValueError("Either iso_surfaces and iso_range or iso_values must be provided")

        # Determine property to compute based on mode
        property_name = self._map_mode_to_property(
            mode, bands, atoms, orbitals, spins, spin_texture
        )

        # Create FermiSurface objects for each energy value
        fermi_surfaces = []
        for e_value in energy_values:
            fs = self._create_fermi_surface(
                fermi=e_value, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins
            )

            # Compute property if needed
            if property_name:
                prop = fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)
                fs.set_values(property_name, prop.value)

            fermi_surfaces.append(fs)

            logger.debug(f"___Generated surface for energy {e_value}___")
            logger.debug(f"Surface has {fs.n_points} points")

        if save_gif is None:
            warn_user("No filename provided for GIF. Setting default filename.")
            save_gif = "fermi_surface.gif"

        # Create plotter and add isovalue gif
        fsplt = FermiPlotter(off_screen=True)

        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_isovalue_gif(
            fermi_surfaces,
            save_gif,
            add_active_vectors=add_active_vectors,
            add_surface_args={
                "show_scalar_bar": config.show_scalar_bar and property_name is not None,
                "cmap": config.surface_cmap,
                "clim": config.surface_clim,
                "opacity": config.surface_opacity,
            },
        )

        user_logger.info(f"GIF saved to {save_gif}")

    def plot_fermi_cross_section(
        self,
        mode,
        slice_normal: tuple[float, float, float] = (1, 0, 0),
        slice_origin: tuple[float, float, float] = (0, 0, 0),
        show_van_alphen_frequency: bool = False,
        show_cross_section_area: bool = False,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
        show=True,
        save_2d=None,
        save_2d_slice=None,
        print_plot_opts: bool = False,
        **kwargs,
    ):
        """A method to plot fermi surface cross sections with an interactive plane slicer

        Parameters
        ----------
        mode : str
            The mode to calculate
        slice_normal : Tuple[float, float, float], optional
            Normal vector of the slicing plane, by default (1, 0, 0)
        slice_origin : Tuple[float, float, float], optional
            Origin point of the slicing plane, by default (0, 0, 0)
        bands : List[int], optional
            A list of band indexes to plot, by default None
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        show : bool, optional
            Whether to show the plot, by default True
        save_2d : str, optional
            Filename to save 2D plot, by default None
        save_2d_slice : str, optional
            Filename to save 2D slice plot, by default None
        print_plot_opts: bool, optional
            Boolean to print the plotting options
        """
        config = ConfigManager.merge_configs(self.default_config, kwargs)
        config = cast(FermiSurface3DConfig, ConfigManager.merge_config(config, "mode", mode))

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Create FermiSurface object
        fermi_surface = self._create_fermi_surface(
            bands=bands, atoms=atoms, orbitals=orbitals, spins=spins
        )

        if fermi_surface.n_points == 0:
            warn_user(
                "No Fermi surface found for the given parameters. Skipping plotting."
            )
            return None

        # Determine and compute property based on mode
        property_name = self._map_mode_to_property(
            mode, bands, atoms, orbitals, spins, spin_texture
        )
        if property_name:
            prop = fermi_surface.get_property(
                property_name, atoms=atoms, orbitals=orbitals, spins=spins
            )
            fermi_surface.set_values(property_name, prop.value)

        # Create plotter and add slicer
        fsplt = _new_plotter(kwargs, save_2d)

        add_active_vectors = spin_texture or property_name == "fermi_velocity"

        fsplt.add_surface(
            fermi_surface,
            add_active_vectors=add_active_vectors,
            show_scalar_bar=False,
            cmap=config.surface_cmap,
            clim=config.surface_clim,
            opacity=config.surface_opacity,
        )

        fsplt.add_slicer(
            fermi_surface,
            normal=slice_normal,
            origin=slice_origin,
            show_van_alphen_frequency=show_van_alphen_frequency,
            show_cross_section_area=show_cross_section_area,
            add_surface_args={
                "show_scalar_bar": config.show_scalar_bar and property_name is not None,
                "add_active_vectors": add_active_vectors,
                "cmap": config.surface_cmap,
                "clim": config.surface_clim,
                "opacity": config.surface_opacity,
            },
        )

        if save_2d:
            fsplt.savefig(filename=save_2d)
        if save_2d_slice:
            fsplt.save_slice_2d(
                fermi_surface,
                slice_normal,
                slice_origin,
                save_2d_slice,
                cmap=config.surface_cmap,
                clim=config.surface_clim,
            )
        if show and not save_2d:
            fsplt.show()

    def plot_fermi_cross_section_box_widget(
        self,
        mode,
        slice_normal: tuple[float, float, float] = (1, 0, 0),
        slice_origin: tuple[float, float, float] = (0, 0, 0),
        show_cross_section_area: bool = False,
        show_van_alphen_frequency: bool = False,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
        show=True,
        save_2d=None,
        save_2d_slice=None,
        print_plot_opts: bool = False,
        show_colorbar: bool = True,
        **kwargs,
    ):
        """A method to plot fermi surface cross sections with box and plane slicing widgets

        Parameters
        ----------
        mode : str
            The mode to calculate
        slice_normal : Tuple[float, float, float], optional
            Normal vector of the slicing plane, by default (1, 0, 0)
        slice_origin : Tuple[float, float, float], optional
            Origin point of the slicing plane, by default (0, 0, 0)
        bands : List[int], optional
            A list of band indexes to plot, by default None
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        show : bool, optional
            Whether to show the plot, by default True
        save_2d : str, optional
            Filename for a screenshot of the 3D view with the widgets. The plot is
            rendered off screen and not shown. By default None
        save_2d_slice : str, optional
            Filename for a matplotlib plot of the cross section at ``slice_normal``
            and ``slice_origin``, by default None
        print_plot_opts: bool, optional
            Boolean to print the plotting options
        """
        config = ConfigManager.merge_configs(self.default_config, kwargs)
        config = cast(FermiSurface3DConfig, ConfigManager.merge_config(config, "mode", mode))

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Create FermiSurface object
        fermi_surface = self._create_fermi_surface(
            bands=bands, atoms=atoms, orbitals=orbitals, spins=spins
        )

        if fermi_surface.n_points == 0:
            warn_user(
                "No Fermi surface found for the given parameters. Skipping plotting."
            )
            return None

        # Determine and compute property based on mode
        property_name = self._map_mode_to_property(
            mode, bands, atoms, orbitals, spins, spin_texture
        )
        if property_name:
            prop = fermi_surface.get_property(
                property_name, atoms=atoms, orbitals=orbitals, spins=spins
            )
            fermi_surface.set_values(property_name, prop.value)

        user_logger.info(f"Generated Fermi surface with {fermi_surface.n_points} points")

        # Create plotter and add box slicer
        fsplt = _new_plotter(kwargs, save_2d)

        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_surface(
            fermi_surface,
            add_active_vectors=add_active_vectors,
            show_scalar_bar=False,
            cmap=config.surface_cmap,
            clim=config.surface_clim,
            opacity=config.surface_opacity,
        )

        fsplt.add_box_slicer(
            fermi_surface,
            normal=slice_normal,
            origin=slice_origin,
            show_cross_section_area=show_cross_section_area,
            show_van_alphen_frequency=show_van_alphen_frequency,
            add_surface_args={
                "show_scalar_bar": config.show_scalar_bar and property_name is not None,
                "add_active_vectors": add_active_vectors,
                "cmap": config.surface_cmap,
                "clim": config.surface_clim,
                "opacity": config.surface_opacity,
            },
        )

        if not (property_name is not None or show_colorbar) or mode == "plain":
            fsplt.remove_scalar_bar()

        if save_2d:
            fsplt.savefig(filename=save_2d)
        if save_2d_slice:
            fsplt.save_slice_2d(
                fermi_surface,
                slice_normal,
                slice_origin,
                save_2d_slice,
                cmap=config.surface_cmap,
                clim=config.surface_clim,
            )
        if show and not save_2d:
            fsplt.show()

    def print_default_settings(self):
        """
        Prints all the configuration settings with their current values.
        """
        for key, value in self.default_config.as_dict().items():
            user_logger.info(f"{key}: {value}")
