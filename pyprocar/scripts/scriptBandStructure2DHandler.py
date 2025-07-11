__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import copy
import logging
import os
import sys
from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import yaml
from matplotlib import cm
from matplotlib import colors as mpcolors

from pyprocar import io

# from pyprocar.fermisurface3d import fermisurface3D
from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.plotter import BandStructure2DataHandler, BandStructure2DVisualizer
from pyprocar.utils import ROOT, data_utils, welcome
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


pv.global_theme.font.size = 10
np.set_printoptions(threshold=sys.maxsize)


class BandStructure2DHandler:

    def __init__(
        self,
        code: str,
        dirname: str = "",
        fermi: float = None,
        fermi_shift: float = 0,
        repair: bool = False,
        apply_symmetry: bool = True,
        use_cache: bool = False,
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
        fermi_shift : float, optional
            The fermi energy shift, by default 0
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

        self.default_config = ConfigFactory.create_config(PlotType.BAND_STRUCTURE_2D)

        modes = ["plain", "parametric", "spin_texture", "overlay"]
        props = ["fermi_speed", "fermi_velocity", "harmonic_effective_mass"]
        modes_txt = " , ".join(modes)
        props_txt = " , ".join(props)
        self.notification_message = f"""
                There are additional plot options that are defined in a configuration file. 
                You can change these configurations by passing the keyword argument to the function
                To print a list of plot options set print_plot_opts=True

                Here is a list modes : {modes_txt}
                Here is a list of properties: {props_txt}
                """

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

        codes_with_scf_fermi = ["qe", "elk"]
        if code in codes_with_scf_fermi and fermi is None:
            logger.info(
                f"No fermi given, using the found fermi energy: {self.ebs.efermi}"
            )

            fermi = self.ebs.efermi

        if fermi is not None:
            logger.info(f"Shifting Fermi energy to zero: {fermi}")

            self.ebs.bands -= fermi
            self.ebs.bands += fermi_shift
            self.fermi_level = fermi_shift
            self.energy_label = r"E - E$_F$ (eV)"
            self.fermi_message = None
        else:
            self.energy_label = r"E (eV)"
            self.fermi_level = None
            self.fermi_message = (
                "`fermi` is not set! Set `fermi={value}`."
                "The plot did not shift the bands by the Fermi energy."
            )

    def process_data(
        self,
        mode,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
    ):
        self.data_handler.process_data(
            mode, bands, atoms, orbitals, spins, spin_texture
        )

    def plot_band_structure(
        self,
        mode,
        bands=None,
        atoms=None,
        orbitals=None,
        spins=None,
        spin_texture=False,
        property_name=None,
        k_z_plane=0,
        k_z_plane_tol=0.0001,
        show=True,
        k_plane_scale=2 * np.pi,
        render_offscreen=False,
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
        render_offscreen: bool, optional
            Boolean to render the plot offscreen, by default False
        save_2d: str, optional
            The path to save the 2d plot, by default None
        save_gif: str, optional
            The path to save the gif, by default None
        save_mp4: str, optional
            The path to save the mp4, by default None
        save_3d: str, optional
            The path to save the 3d plot, by default None
        """

        config = ConfigManager.merge_configs(self.default_config, kwargs)
        config = ConfigManager.merge_config(config, "mode", mode)

        user_logger.info("_" * 100)
        user_logger.info(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        user_logger.info("_" * 100)

        if self.fermi_message:
            user_logger.info(self.fermi_message)
        user_logger.warning(
            f"Make sure the kmesh has kz points with kz={k_z_plane} +- {k_z_plane_tol}"
        )

        # Process the data
        self.ebs.reduce_bands_near_fermi(bands=bands, tolerance=0.7)
        self.ebs.expand_kpoints_to_supercell()
        self.ebs.reduce_kpoints_to_plane(k_z_plane, k_z_plane_tol)
        self.data_handler = BandStructure2DataHandler(self.ebs, config=config)

        self.data_handler.process_data(
            mode,
            bands=bands,
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            spin_texture=spin_texture,
        )
        band_structure_surface = self.data_handler.get_surface_data(
            property_name=property_name
        )
        visualizer = BandStructure2DVisualizer(self.data_handler, config=config)
        visualizer.plotter.off_screen = render_offscreen

        k_plane_scale_transform = np.eye(4)

        k_plane_scale_transform[0, 0] = k_plane_scale * k_plane_scale_transform[0, 0]
        k_plane_scale_transform[1, 1] = k_plane_scale * k_plane_scale_transform[1, 1]
        band_structure_surface.transform(k_plane_scale_transform)

        band_structure_surface.brillouin_zone.transform(k_plane_scale_transform)
        if config.show_brillouin_zone:
            visualizer.add_brillouin_zone(band_structure_surface)

        if config.clip_brillouin_zone:
            band_structure_surface = visualizer.clip_brillouin_zone(
                band_structure_surface
            )

        if (
            visualizer.data_handler.scalars_name == "spin_magnitude"
            or visualizer.data_handler.scalars_name == "Band Velocity Vector_magnitude"
        ):
            visualizer.add_texture(
                band_structure_surface,
                scalars_name=visualizer.data_handler.scalars_name,
                vector_name=visualizer.data_handler.vector_name,
            )

        visualizer.add_surface(band_structure_surface)

        if (mode != "plain" or spin_texture) and config.show_scalar_bar:
            visualizer.add_scalar_bar(name=visualizer.data_handler.scalars_name)

        if config.show_grid:
            visualizer.add_grid(z_label=self.energy_label)

        if config.show_axes:
            visualizer.add_axes()

        if self.fermi_level is not None:
            visualizer.add_fermi_plane(value=self.fermi_level)

        visualizer.set_background_color()

        if save_2d:
            visualizer.savefig(filename=save_2d)
            return None

            # save and showing setting
        if show and (save_gif is None and save_mp4 is None and save_3d is None):
            user_message = visualizer.show()
            user_logger.info(user_message)

        if save_gif:
            visualizer.save_gif(filename=save_gif, **config.save_gif_config)
        if save_mp4:
            visualizer.save_mp4(filename=save_mp4, **config.save_mp4_config)
        if save_3d:
            visualizer.save_mesh(
                filename=save_3d,
                surface=band_structure_surface,
                **config.save_mesh_config,
            )

        visualizer.close()

    def print_default_settings(self):
        """
        Prints all the configuration settings with their current values.
        """
        for key, value in self.default_config.as_dict().items():
            user_logger.info(f"{key}: {value}")


# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx
