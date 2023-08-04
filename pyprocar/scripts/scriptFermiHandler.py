__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import sys
import functools
import copy
from typing import List, Tuple
import os
import yaml

import numpy as np
from matplotlib import colors as mpcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import pyvista as pv

from pyprocar.core import FermiSurface3D
from pyprocar.plotter import FermiDataHandler, FermiVisualizer
from pyprocar.utils import ROOT

from .. import io

np.set_printoptions(threshold=sys.maxsize)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
class FermiHandler:

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
        self.e_fermi = parser.ebs.efermi
        self.structure = parser.structure
        if self.structure.rotations is not None:
            self.ebs.ibz2fbz(self.structure.rotations)

        self.data_handler = FermiDataHandler(self.ebs)

    def process_data(self, mode, bands=None, atoms=None, orbitals=None, spins=None, spin_texture=False):
        self.data_handler.process_data(mode, bands, atoms, orbitals, spins, spin_texture)

    def plot_fermi_surface(self, mode, 
                           bands=None, 
                           atoms=None, 
                           orbitals=None, 
                           spins=None, 
                           spin_texture=False,
                           property_name=None,
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
        # Process the data
        self.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
        fermi_surface=self.data_handler.get_surface_data(fermi=self.e_fermi,property_name=property_name)

        visualizer = FermiVisualizer(self.data_handler,**kwargs)
        visualizer.add_brillouin_zone(fermi_surface)

        visualizer.add_texture(
                    fermi_surface,
                    scalars_name=visualizer.data_handler.scalars_name, 
                    vector_name=visualizer.data_handler.vector_name)
        visualizer.add_surface(fermi_surface)
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
            visualizer.save_mesh(filename=save_3d,surface=fermi_surface)

    def plot_fermi_isoslider(self, mode, 
                            iso_range: float=None,
                            iso_surfaces: int=None,
                            iso_values: List[float]=None,
                            bands=None, 
                            atoms=None, 
                            orbitals=None, 
                            spins=None, 
                            spin_texture=False,
                            property_name=None,
                            show=True,
                            save_2d=None,
                            **kwargs):
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
        """
        # Process the data
        self.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
        
        if iso_surfaces is not None:
            energy_values = np.linspace(self.e_fermi-iso_range/2,self.e_fermi+iso_range/2,iso_surfaces)
        if iso_values:
            energy_values=iso_values


        e_surfaces = []
        for e_value in energy_values:
            surface=self.data_handler.get_surface_data(property_name=property_name,fermi=e_value)
            e_surfaces.append(surface)

        visualizer = FermiVisualizer(self.data_handler,**kwargs)
        
        visualizer.add_isovalue(e_surfaces,energy_values)


        # save and showing setting
        if show:
            visualizer.show(filename=save_2d)

    def create_isovalue_gif(self, mode, 
                            iso_range: float=None,
                            iso_surfaces: int=None,
                            iso_values: List[float]=None,
                            bands=None, 
                            atoms=None, 
                            orbitals=None, 
                            spins=None, 
                            spin_texture=False,
                            property_name=None,
                            show=True,
                            save_gif=None,
                            **kwargs):
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
        """
        # Process the data
        self.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
        
        if iso_surfaces is not None:
            energy_values = np.linspace(self.e_fermi-iso_range/2,self.e_fermi+iso_range/2,iso_surfaces)
        if iso_values:
            energy_values=iso_values


        e_surfaces = []
        for e_value in energy_values:
            surface=self.data_handler.get_surface_data(property_name=property_name,fermi=e_value)
            e_surfaces.append(surface)

        visualizer = FermiVisualizer(self.data_handler,**kwargs)
        
        visualizer.add_isovalue_gif(e_surfaces,energy_values,save_gif)

    def plot_fermi_cross_section(self,
                            mode,
                            slice_normal: Tuple[float,float,float]=(1,0,0),
                            slice_origin: Tuple[float,float,float]=(0,0,0),
                            bands=None, 
                            atoms=None, 
                            orbitals=None, 
                            spins=None, 
                            spin_texture=False,
                            property_name=None,
                            show=True,
                            save_2d=None,
                            save_2d_slice=None,
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
        # Process the data
        self.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
        surface=self.data_handler.get_surface_data(fermi=self.e_fermi,property_name=property_name)

        visualizer = FermiVisualizer(self.data_handler,**kwargs)
        visualizer.add_slicer(surface,show,save_2d,save_2d_slice,slice_normal,slice_origin)

    def plot_fermi_cross_section_box_widget(self,
                            mode,
                            slice_normal: Tuple[float,float,float]=(1,0,0),
                            slice_origin: Tuple[float,float,float]=(0,0,0),
                            bands=None, 
                            atoms=None, 
                            orbitals=None, 
                            spins=None, 
                            spin_texture=False,
                            property_name=None,
                            show=True,
                            save_2d=None,
                            save_2d_slice=None,
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
        # Process the data
        self.process_data(mode, bands=bands, atoms=atoms, orbitals=orbitals, spins=spins, spin_texture=spin_texture)
        surface=self.data_handler.get_surface_data(fermi=self.e_fermi,property_name=property_name)

        visualizer = FermiVisualizer(self.data_handler,**kwargs)
        visualizer.add_box_slicer(surface,show,save_2d,save_2d_slice,slice_normal,slice_origin)


    
    def default_settings(self):
        with open(os.path.join(ROOT,'pyprocar','cfg','fermi_surface_3d.yml'), 'r') as file:
            plotting_options = yaml.safe_load(file)
        for key,value in plotting_options.items():
                print(key,':',value)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx