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

from pyprocar import io
from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.plotter import FermiDataHandler, FermiVisualizer
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
        repair: bool = False,
        apply_symmetry: bool = True,
        ebs_interpolation_factor=1,
        use_cache: bool = False,
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
        """


        welcome()

        user_logger.info("_" * 100)

        self.default_config = ConfigFactory.create_config(PlotType.FERMI_SURFACE_3D)

        self.code = code
        self.dirname = dirname
        self.repair = repair
        self.apply_symmetry = apply_symmetry

        ebs_pkl_filepath = os.path.join(dirname, "ebs.pkl")
        structure_pkl_filepath = os.path.join(dirname, "structure.pkl")

        if not use_cache:
            if os.path.exists(structure_pkl_filepath):
                logger.info(
                    f"Removing existing structure file: {structure_pkl_filepath}"
                )
                os.remove(structure_pkl_filepath)
            if os.path.exists(ebs_pkl_filepath):
                logger.info(f"Removing existing EBS file: {ebs_pkl_filepath}")
                os.remove(ebs_pkl_filepath)

        if not os.path.exists(ebs_pkl_filepath):
            logger.info(f"Parsing EBS from {dirname}")

            parser = io.Parser(code=code, dirpath=dirname)
            ebs = parser.ebs
            structure = parser.structure

            if structure.rotations is not None:
                logger.info(
                    f"Detected symmetry operations ({structure.rotations.shape})."
                    " Applying to ebs to get full BZ"
                )
                ebs.ibz2fbz(structure.rotations)

            data_utils.save_pickle(ebs, ebs_pkl_filepath)
            data_utils.save_pickle(structure, structure_pkl_filepath)
        else:
            logger.info(
                f"Loading EBS and Structure from cached Pickle files in {dirname}"
            )

            ebs = data_utils.load_pickle(ebs_pkl_filepath)
            structure = data_utils.load_pickle(structure_pkl_filepath)

        self.ebs = ebs
        self.structure = structure

        if fermi is None:
            self.e_fermi = self.ebs.efermi
            user_logger.warning(
                f"Fermi Energy not set! Set `fermi={self.e_fermi}`."
                "By default, using fermi energy found in the current directory."
            )
        else:
            self.e_fermi = fermi

        if ebs_interpolation_factor != 1:
            self.ebs = self.ebs.interpolate(
                interpolation_factor=ebs_interpolation_factor
            )

        modes = ["plain", "parametric", "spin_texture", "overlay"]
        props = ["fermi_speed", "fermi_velocity", "harmonic_effective_mass"]
        modes_txt = " , ".join(modes)
        props_txt = " , ".join(props)
        self.notification_message = f"""
                There are additional plot options that are defined in a configuration file. 
                You can change these configurations by passing the keyword argument to the function
                To print a list of plot options set print_plot_opts=True

                Here is a list modes : {modes_txt}
                Here is a list of properties: {props_txt}"""

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

        # Process the data
        self.data_handler = FermiDataHandler(self.ebs, config)
        self.data_handler.process_data(
            bands=bands,
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            spin_texture=spin_texture,
        )
        fermi_surface = self.data_handler.get_surface_data(
            fermi=self.e_fermi,
            property_name=config.property_name,
            fermi_shift=fermi_shift,
        )
        if fermi_surface is None:
            user_logger.warning(
                f"No Fermi surface found for spin {spins}. Skipping plotting."
            )
            return None

        visualizer = FermiVisualizer(self.data_handler, config)

        if config.show_brillouin_zone:
            visualizer.add_brillouin_zone(fermi_surface)

        if (
            visualizer.data_handler.scalars_name == "spin_magnitude"
            or visualizer.data_handler.scalars_name == "Fermi Velocity Vector_magnitude"
        ):
            visualizer.add_texture(
                fermi_surface,
                scalars_name=visualizer.data_handler.scalars_name,
                vector_name=visualizer.data_handler.vector_name,
            )

        visualizer.add_surface(fermi_surface)

        if (mode != "plain" or spin_texture) and config.show_scalar_bar:
            visualizer.add_scalar_bar(name=visualizer.data_handler.scalars_name)

        if config.show_axes:
            visualizer.add_axes()

        visualizer.set_background_color()

        if save_2d:
            visualizer.savefig(filename=save_2d)
            return None

        # save and showing setting
        if show and (save_gif is None and save_mp4 is None and save_3d is None):
            user_message = visualizer.show()
            user_logger.info(user_message)

        if save_gif is not None:
            visualizer.save_gif(
                filename=save_gif, save_gif_config=config.save_gif_config
            )
        if save_mp4:
            visualizer.save_mp4(
                filename=save_mp4, save_mp4_config=config.save_mp4_config
            )
        if save_3d:
            visualizer.save_mesh(
                filename=save_3d,
                surface=fermi_surface,
                save_mesh_config=config.save_mesh_config,
            )

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
        """A method to plot the 3d fermi surface

        Parameters
        ----------
        iso_range : float
            A range of energies the slide will go through
        iso_surfaces : int
            Ther number of fermi sruface to calculate on the range
        iso_range : List[int], optional
            A list of of energies the slider will go through
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

        # Process the data
        self.data_handler = FermiDataHandler(self.ebs, config)
        self.data_handler.process_data(
            bands=bands,
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            spin_texture=spin_texture,
        )

        if iso_surfaces is not None:
            energy_values = np.linspace(
                self.e_fermi - iso_range / 2, self.e_fermi + iso_range / 2, iso_surfaces
            )
        if iso_values:
            energy_values = iso_values

        e_surfaces = []
        for e_value in energy_values:
            surface = self.data_handler.get_surface_data(
                property_name=config.property_name, fermi=e_value
            )
            e_surfaces.append(surface)

            logger.debug(f"___Getting surface for {e_value}__")
            logger.debug(f"Surface shape: {surface.points.shape}")
            logger.debug(f"Surface shape: {surface.point_data}")
            logger.debug(f"Surface shape: {surface.point_data}")

        visualizer = FermiVisualizer(self.data_handler, config)

        visualizer.add_isoslider(e_surfaces, energy_values)

        # save and showing setting
        if save_2d:
            visualizer.savefig(filename=save_2d)
            return None

        # save and showing setting
        if show:
            user_message = visualizer.show()
            user_logger.info(user_message)

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
        """A method to plot the 3d fermi surface

        Parameters
        ----------
        iso_range : float
            A range of energies the slide will go through
        iso_surfaces : int
            Ther number of fermi sruface to calculate on the range
        iso_range : List[int], optional
            A list of of energies the slider will go through
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

        # Process the data
        self.data_handler = FermiDataHandler(self.ebs, config)
        self.data_handler.process_data(
            bands=bands,
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            spin_texture=spin_texture,
        )

        if iso_surfaces is not None:
            energy_values = np.linspace(
                self.e_fermi - iso_range / 2, self.e_fermi + iso_range / 2, iso_surfaces
            )
        if iso_values:
            energy_values = iso_values

        e_surfaces = []
        for e_value in energy_values:
            surface = self.data_handler.get_surface_data(
                fermi=e_value, property_name=config.property_name
            )
            logger.debug(f"___Getting surface for {e_value}__")
            logger.debug(f"Surface shape: {surface.points.shape}")
            logger.debug(f"Surface shape: {surface.point_data}")
            logger.debug(f"Surface shape: {surface.point_data}")
            e_surfaces.append(surface)

        visualizer = FermiVisualizer(self.data_handler, config)

        visualizer.add_isovalue_gif(e_surfaces, energy_values, save_gif)

    def plot_fermi_cross_section(
        self,
        mode,
        slice_normal: Tuple[float, float, float] = (1, 0, 0),
        slice_origin: Tuple[float, float, float] = (0, 0, 0),
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

        # Process the data
        self.data_handler = FermiDataHandler(self.ebs, config)
        self.data_handler.process_data(
            bands=bands,
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            spin_texture=spin_texture,
        )
        surface = self.data_handler.get_surface_data(
            fermi=self.e_fermi, property_name=config.property_name
        )

        visualizer = FermiVisualizer(self.data_handler, config)
        visualizer.add_slicer(
            surface, show, save_2d, save_2d_slice, slice_normal, slice_origin
        )

    def plot_fermi_cross_section_box_widget(
        self,
        mode,
        slice_normal: Tuple[float, float, float] = (1, 0, 0),
        slice_origin: Tuple[float, float, float] = (0, 0, 0),
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

        # Process the data
        self.data_handler = FermiDataHandler(self.ebs, config)
        self.data_handler.process_data(
            bands=bands,
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            spin_texture=spin_texture,
        )
        surface = self.data_handler.get_surface_data(
            fermi=self.e_fermi, property_name=config.property_name
        )
        user_logger.info(
            "Bands being used if bands=None: ", surface.band_isosurface_index_map
        )
        visualizer = FermiVisualizer(self.data_handler, config)
        visualizer.add_box_slicer(
            surface, show, save_2d, save_2d_slice, slice_normal, slice_origin
        )

    def print_default_settings(self):
        """
        Prints all the configuration settings with their current values.
        """
        for key, value in self.default_config.as_dict().items():
            user_logger.info(f"{key}: {value}")
