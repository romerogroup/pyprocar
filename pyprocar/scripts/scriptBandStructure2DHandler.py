__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import sys
import copy
from typing import List, Tuple
import os
import yaml
from itertools import product

import numpy as np
from matplotlib import colors as mpcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import pyvista as pv

# from pyprocar.fermisurface3d import fermisurface3D
from pyprocar.plotter import BandStructure2DataHandler, BandStructure2DVisualizer
from pyprocar.utils import ROOT
from .. import io

pv.global_theme.font.size = 10
np.set_printoptions(threshold=sys.maxsize)
# 
class BandStructure2DHandler:

    def __init__(self, 
            code:str,
            dirname:str="",
            fermi:float=None,
            fermi_shift:float=0,
            repair:bool=False,
            apply_symmetry:bool=True,):
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
        """


        modes=["plain","parametric","spin_texture", "overlay" ]
        props=["fermi_speed","fermi_velocity","harmonic_effective_mass"]
        modes_txt=' , '.join(modes)
        props_txt=' , '.join(props)
        self.notification_message=f"""
                ----------------------------------------------------------------------------------------------------------
                There are additional plot options that are defined in a configuration file. 
                You can change these configurations by passing the keyword argument to the function
                To print a list of plot options set print_plot_opts=True

                Here is a list modes : {modes_txt}
                Here is a list of properties: {props_txt}
                ----------------------------------------------------------------------------------------------------------
                """
        

        self.code = code
        self.dirname=dirname
        self.repair = repair
        self.apply_symmetry = apply_symmetry
        
        parser = io.Parser(code = code, dir = dirname)
        self.ebs = parser.ebs

        codes_with_scf_fermi = ['qe', 'elk']
        if code in codes_with_scf_fermi and fermi is None:
            fermi = self.ebs.efermi
        if fermi is not None:
            self.ebs.bands -= fermi
            self.ebs.bands += fermi_shift
            self.fermi_level = fermi_shift
            self.energy_label=r"E - E$_F$ (eV)"
            self.fermi_message=None
        else:
            self.energy_label=r"E (eV)"
            self.fermi_level=None
            self.fermi_message="""
                WARNING : `fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy.
                ----------------------------------------------------------------------------------------------------------
                """

        # Applying symmetry to kmesh if they exists
        self.structure = parser.structure
        if self.structure.rotations is not None:
            self.ebs.ibz2fbz(self.structure.rotations)

    def process_data(self, mode, bands=None, atoms=None, orbitals=None, spins=None, spin_texture=False):
        self.data_handler.process_data(mode, bands, atoms, orbitals, spins, spin_texture)

    def plot_band_structure(self, mode, 
                            bands=None, 
                            atoms=None, 
                            orbitals=None, 
                            spins=None, 
                            spin_texture=False,
                            property_name=None,
                            k_z_plane=0, 
                            k_z_plane_tol=0.0001,
                            show=True,
                            save_2d=None,
                            save_gif=None,
                            save_mp4=None,
                            save_3d=None,
                            print_plot_opts:bool=False,
                            **kwargs):
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
        print(self.notification_message)
        if print_plot_opts:
            self.print_default_settings()
        if self.fermi_message:
            print(self.fermi_message)
        print(
        f"""
            WARNING : Make sure the kmesh has kz points with kz={k_z_plane} +- {k_z_plane_tol}
            ----------------------------------------------------------------------------------------------------------
            """)

        # Process the data
        self.ebs.reduce_bands_near_fermi(bands=bands, tolerance=0.7)
        self.ebs.expand_kpoints_to_supercell()
        self.ebs.reduce_kpoints_to_plane(k_z_plane,k_z_plane_tol)
        self.data_handler = BandStructure2DataHandler(self.ebs, **kwargs)
        # self.ebs.reduce_kpoints_to_plane(k_z_plane,k_z_plane_tol)
        # self.data_handler.ebs.reduce_kpoints_to_plane(k_z_plane,k_z_plane_tol)

        self.data_handler.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
        band_structure_surface=self.data_handler.get_surface_data(property_name=property_name)
        visualizer = BandStructure2DVisualizer(self.data_handler,**kwargs)
        visualizer.add_brillouin_zone(band_structure_surface)
        band_structure_surface=visualizer.clip_broullin_zone(band_structure_surface)
        visualizer.add_texture(
                    band_structure_surface,
                    scalars_name=visualizer.data_handler.scalars_name, 
                    vector_name=visualizer.data_handler.vector_name)
        visualizer.add_surface(band_structure_surface)
        if mode != "plain" or spin_texture:
            visualizer.add_scalar_bar(name=visualizer.data_handler.scalars_name)
        
        visualizer.add_grid(z_label=self.energy_label)
        visualizer.add_axes()

        if self.fermi_level is not None:
            visualizer.add_fermi_plane(value=self.fermi_level)

        visualizer.set_background_color()
        
        # save and showing setting
        if show and save_gif is None and save_mp4 is None and save_3d is None:
            visualizer.show(filename=save_2d)
        if save_gif is not None:
            visualizer.save_gif(filename=save_gif)
        if save_mp4:
            visualizer.save_gif(filename=save_mp4)
        if save_3d:
            visualizer.save_mesh(filename=save_3d,surface=band_structure_surface)

        visualizer.close()

    def print_default_settings(self):
        with open(os.path.join(ROOT,'pyprocar','cfg','band_structure_2d.yml'), 'r') as file:
            plotting_options = yaml.safe_load(file)
        
        for key,value in plotting_options.items():
            print(key,':',value)


# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx