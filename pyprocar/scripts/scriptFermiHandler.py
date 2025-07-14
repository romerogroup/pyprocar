__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import yaml

from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.io import Parser
from pyprocar.plotter import FermiPlotter
from pyprocar.utils import ROOT, data_utils, welcome
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



class FermiHandler:

    def __init__(
        self,
        code: str,
        dirname: str = "",
        fermi: float = None,
        ebs_interpolation_factor=1,
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
        repair : bool, optional
            Boolean to repair the PROCAR file, by default False
        apply_symmetry : bool, optional
            Boolean to apply symmetry to the fermi sruface.
            This is used when only symmetry reduced kpoints used in the calculation, by default True
        use_cache : bool, optional
            Boolean to use cached Pickle files, by default True
        verbose : int, optional
            Verbosity level, by default 1
        """
        set_verbose_level(verbose)

        user_logger.info(f"If you want more detailed logs, set verbose to 2 or more")
        user_logger.info("_" * 100)

        welcome()

        user_logger.info("_" * 100)

        self.default_config = ConfigFactory.create_config(PlotType.FERMI_SURFACE_3D)
        
        # Store parameters for creating FermiSurface objects
        self.code = code
        self.dirname = dirname
        self.ebs_interpolation_factor = ebs_interpolation_factor
        
        # Create a sample FermiSurface to get default fermi energy
        sample_fs = FermiSurface.from_code(code, dirname)
        
        if fermi is None:
            self.e_fermi = sample_fs.fermi
            user_logger.warning(
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

    def _map_mode_to_property(self, mode, bands=None, atoms=None, orbitals=None, spins=None, spin_texture=False):
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
            user_logger.warning(f"Unknown mode: {mode}. Using plain mode.")
            return None
    
    def _create_fermi_surface(self, fermi=None, fermi_shift=0.0, bands=None, atoms=None, orbitals=None, spins=None):
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
            
        fermi_surface = FermiSurface.from_code(
            self.code, 
            self.dirname, 
            fermi=fermi, 
            fermi_shift=fermi_shift
        )
        
        return fermi_surface

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
        config = ConfigManager.merge_config(config, "mode", mode)

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Create FermiSurface object
        fermi_surface = self._create_fermi_surface(fermi_shift=fermi_shift, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins)

        if fermi_surface.n_points == 0:
            user_logger.warning(
                f"No Fermi surface found for the given parameters. Skipping plotting."
            )
            return None

        # Determine and compute property based on mode
        property_name = self._map_mode_to_property(mode, bands, atoms, orbitals, spins, spin_texture)
        if property_name and mode != "plain":
            fermi_surface.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)

        # Create plotter and add components
        fsplt = FermiPlotter(**{k:v for k,v in kwargs.items() if k in ['off_screen', 'window_size', 'theme']})
        
        if config.show_brillouin_zone:
            fsplt.add_brillouin_zone(fermi_surface.brillouin_zone)

        # Add the surface with appropriate settings
        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_surface(
            fermi_surface, 
            show_scalar_bar=(property_name is not None or show_colorbar),
            scalars=property_name if mode == "plain" else None,
            add_active_vectors=add_active_vectors,
            cmap=config.surface_cmap,
            clim=config.surface_clim,
            opacity=config.surface_opacity
        )
        
        if not (property_name is not None or show_colorbar) or mode == "plain":
            fsplt.remove_scalar_bar()

        if config.show_axes:
            fsplt.add_axes(
                xlabel=config.x_axes_label,
                ylabel=config.y_axes_label,
                zlabel=config.z_axes_label,
            )

        # Handle saving and showing
        if save_2d:
            fsplt.savefig(filename=save_2d)
            return None

        if show and (save_gif is None and save_mp4 is None and save_3d is None):
            fsplt.show()

        if save_gif is not None:
            user_logger.warning("GIF saving not yet implemented in new API")
            
        if save_mp4:
            user_logger.warning("MP4 saving not yet implemented in new API")
            
        if save_3d:
            user_logger.warning("3D mesh saving not yet implemented in new API")

    def plot_fermi_isoslider(
        self,
        mode,
        iso_range: float = None,
        iso_surfaces: int = None,
        iso_values: List[float] = None,
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
        property_name = self._map_mode_to_property(mode, bands, atoms, orbitals, spins, spin_texture)
        
        # Create FermiSurface objects for each energy value
        fermi_surfaces = []
        for e_value in energy_values:
            fs = self._create_fermi_surface(fermi=e_value, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins)
            
            # Compute property if needed
            if property_name:
                fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)
            
            fermi_surfaces.append(fs)

            logger.debug(f"___Generated surface for energy {e_value}___")
            logger.debug(f"Surface has {fs.n_points} points")

        # Create plotter and add isoslider
        fsplt = FermiPlotter(**{k:v for k,v in kwargs.items() if k in ['off_screen', 'window_size', 'theme']})
        
        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_isoslider(
            fermi_surfaces, 
            energy_values,
            add_active_vectors=add_active_vectors,
            add_surface_args={
                'show_scalar_bar': config.show_scalar_bar and property_name is not None,
                'cmap': config.surface_cmap,
                'clim': config.surface_clim,
                'opacity': config.surface_opacity
            }
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
        iso_values: List[float] = None,
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
        property_name = self._map_mode_to_property(mode, bands, atoms, orbitals, spins, spin_texture)
        
        # Create FermiSurface objects for each energy value
        fermi_surfaces = []
        for e_value in energy_values:
            fs = self._create_fermi_surface(fermi=e_value, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins)
            
            # Compute property if needed
            if property_name:
                fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)
            
            fermi_surfaces.append(fs)

            logger.debug(f"___Generated surface for energy {e_value}___")
            logger.debug(f"Surface has {fs.n_points} points")

        if save_gif is None:
            user_logger.warning("No filename provided for GIF. Setting default filename.")
            save_gif = "fermi_surface.gif"

        # Create plotter and add isovalue gif
        fsplt = FermiPlotter(off_screen=True)
        
        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_isovalue_gif(
            fermi_surfaces, 
            save_gif,
            add_active_vectors=add_active_vectors,
            add_surface_args={
                'show_scalar_bar': config.show_scalar_bar and property_name is not None,
                'cmap': config.surface_cmap,
                'clim': config.surface_clim,
                'opacity': config.surface_opacity
            }
        )
        
        user_logger.info(f"GIF saved to {save_gif}")

    def plot_fermi_cross_section(
        self,
        mode,
        slice_normal: Tuple[float, float, float] = (1, 0, 0),
        slice_origin: Tuple[float, float, float] = (0, 0, 0),
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
        config = ConfigManager.merge_config(config, "mode", mode)

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        # Create FermiSurface object
        fermi_surface = self._create_fermi_surface(bands=bands, atoms=atoms, orbitals=orbitals, spins=spins)

        if fermi_surface.n_points == 0:
            user_logger.warning(
                f"No Fermi surface found for the given parameters. Skipping plotting."
            )
            return None

        # Determine and compute property based on mode
        property_name = self._map_mode_to_property(mode, bands, atoms, orbitals, spins, spin_texture)
        if property_name:
            fermi_surface.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)

        # Create plotter and add slicer
        fsplt = FermiPlotter(**{k:v for k,v in kwargs.items() if k in ['off_screen', 'window_size', 'theme']})
        
        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        
        fsplt.add_surface(fermi_surface,
                          add_active_vectors=add_active_vectors,
                          show_scalar_bar=False,
                          cmap=config.surface_cmap,
                          clim=config.surface_clim,
                          opacity=config.surface_opacity)
        
        fsplt.add_slicer(
            fermi_surface,
            normal=slice_normal,
            origin=slice_origin,
            show_van_alphen_frequency=show_van_alphen_frequency,
            show_cross_section_area=show_cross_section_area,
            add_surface_args={
                'show_scalar_bar': config.show_scalar_bar and property_name is not None,
                'add_active_vectors': add_active_vectors,
                'cmap': config.surface_cmap,
                'clim': config.surface_clim,
                'opacity': config.surface_opacity
            }
        )

        # Handle saving and showing
        if save_2d:
            fsplt.savefig(filename=save_2d)
            return None

        if show:
            fsplt.show()
            
        if save_2d_slice:
            user_logger.warning("2D slice saving not yet implemented in new API")

    def plot_fermi_cross_section_box_widget(
        self,
        mode,
        slice_normal: Tuple[float, float, float] = (1, 0, 0),
        slice_origin: Tuple[float, float, float] = (0, 0, 0),
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
            Filename to save 2D plot, by default None
        save_2d_slice : str, optional
            Filename to save 2D slice plot, by default None
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

        # Create FermiSurface object
        fermi_surface = self._create_fermi_surface(bands=bands, atoms=atoms, orbitals=orbitals, spins=spins)

        if fermi_surface.n_points == 0:
            user_logger.warning(
                f"No Fermi surface found for the given parameters. Skipping plotting."
            )
            return None

        # Determine and compute property based on mode
        property_name = self._map_mode_to_property(mode, bands, atoms, orbitals, spins, spin_texture)
        if property_name:
            fermi_surface.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)

        user_logger.info(f"Generated Fermi surface with {fermi_surface.n_points} points")

        # Create plotter and add box slicer
        fsplt = FermiPlotter(**{k:v for k,v in kwargs.items() if k in ['off_screen', 'window_size', 'theme']})
        
        add_active_vectors = spin_texture or property_name == "fermi_velocity"
        fsplt.add_surface(fermi_surface,
                          add_active_vectors=add_active_vectors,
                          show_scalar_bar=False,
                          cmap=config.surface_cmap,
                          clim=config.surface_clim,
                          opacity=config.surface_opacity)
        
        fsplt.add_box_slicer(
            fermi_surface,
            normal=slice_normal,
            origin=slice_origin,
            save_2d=save_2d,
            save_2d_slice=save_2d_slice,
            show_cross_section_area=show_cross_section_area,
            show_van_alphen_frequency=show_van_alphen_frequency,
            add_surface_args={
                'show_scalar_bar': config.show_scalar_bar and property_name is not None,
                'add_active_vectors': add_active_vectors,
                'cmap': config.surface_cmap,
                'clim': config.surface_clim,
                'opacity': config.surface_opacity
            }
        )
        
        if not (property_name is not None or show_colorbar) or mode == "plain":
            fsplt.remove_scalar_bar()

        # Handle saving and showing
        if show:
            fsplt.show()

    def print_default_settings(self):
        """
        Prints all the configuration settings with their current values.
        """
        for key, value in self.default_config.as_dict().items():
            user_logger.info(f"{key}: {value}")
