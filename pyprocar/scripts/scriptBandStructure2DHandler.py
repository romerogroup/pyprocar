__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import sys
import copy
from typing import List, Tuple
import os
import yaml

import numpy as np
from matplotlib import colors as mpcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import pyvista as pv

# from pyprocar.fermisurface3d import fermisurface3D
from pyprocar.plotter import BandStructure2DataHandler, BandStructure2DVisualizer
from pyprocar.utils import ROOT
from .. import io


np.set_printoptions(threshold=sys.maxsize)

class BandStructure2DHandler:

    def __init__(self, 
            code:str,
            dirname:str="",
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
        repair : bool, optional
            Boolean to repair the PROCAR file, by default False
        apply_symmetry : bool, optional
            Boolean to apply symmetry to the fermi sruface.
            This is used when only symmetry reduced kpoints used in the calculation, by default True
        """
        self.code = code
        self.dirname=dirname
        self.repair = repair
        self.apply_symmetry = apply_symmetry
        parser = io.Parser(code = code, dir = dirname)
        self.ebs = parser.ebs
        self.e_fermi=self.ebs.efermi
        self.structure = parser.structure
        if self.structure.rotations is not None:
            self.ebs.ibz2fbz(self.structure.rotations)

        self._find_bands_near_fermi()
        self.data_handler = BandStructure2DataHandler(self.ebs)

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
        """
        self._reduce_kpoints_to_plane(k_z_plane,k_z_plane_tol)
        # Process the data
        self.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
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

        visualizer.add_axes()
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

    def default_settings(self):
        with open(os.path.join(ROOT,'pyprocar','cfg','fermi_surface_3d.yml'), 'r') as file:
            plotting_options = yaml.safe_load(file)
        for key,value in plotting_options.items():
                print(key,':',value)

    def _find_bands_near_fermi(self,energy_tolerance=0.7):
        energy_level = 0
        full_band_index = []
        for iband in range(len(self.ebs.bands[0,:,0])):
            fermi_surface_test = len(np.where(np.logical_and(self.ebs.bands[:,iband,0]>=energy_level-energy_tolerance, self.ebs.bands[:,iband,0]<=energy_level+energy_tolerance))[0])
            if fermi_surface_test != 0:
                full_band_index.append(iband)
        
        self.ebs.bands=self.ebs.bands[:,full_band_index,:]
        self.ebs.projected=self.ebs.projected[:,full_band_index,:,:,:]
        print("Bands near the fermi level : " , full_band_index )

    def _reduce_kpoints_to_plane(self,k_z_plane,k_z_plane_tol):
        i_kpoints_near_z_0 = np.where(np.logical_and(self.ebs.kpoints_cartesian[:,2] < k_z_plane + k_z_plane_tol, 
                                                     self.ebs.kpoints_cartesian[:,2] > k_z_plane - k_z_plane_tol) )
        self.ebs.kpoints = self.ebs.kpoints[i_kpoints_near_z_0,:][0]
        self.ebs.bands = self.ebs.bands[i_kpoints_near_z_0,:][0]
        self.ebs.projected = self.ebs.projected[i_kpoints_near_z_0,:][0]

# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx