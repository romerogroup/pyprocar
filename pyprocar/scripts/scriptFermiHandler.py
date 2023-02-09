__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import sys
import functools
import copy
from typing import List, Tuple
import os

import numpy as np
from matplotlib import colors as mpcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import vtk
import pyvista as pv
from pyvista.utilities import NORMALS, generate_plane, get_array, try_callback

# from pyprocar.fermisurface3d import fermisurface3D

from ..core import FermiSurface3D
from ..splash import welcome
from ..utilsprocar import UtilsProcar
from ..io.procarparser import ProcarParser
from ..procarselect import ProcarSelect

from ..io.lobsterparser import LobsterFermiParser
from ..io.abinitparser import AbinitParser
from .. import io

np.set_printoptions(threshold=sys.maxsize)



class FermiHandler:
    """
    This class handles the plotting of the fermi surface. Initialize by specifying the code and directory name where the data is stored. 
    Then call one of the plotting methods provided.
    """
    def __init__(self,
            code:str,
            dirname:str="",
            repair:bool=False,
            apply_symmetry:bool=True,
        ):
        self.code = code
        self.dirname=dirname
        self.repair = repair
        self.apply_symmetry = apply_symmetry

        self.parser, self.reciprocal_lattice, self.e_fermi = self.__parse_code()

    def plot_fermi_surface(self, 
                        mode:str,
                        bands:List[int]=None, 
                        atoms:List[int]=None,
                        orbitals:List[int]=None,
                        spins:List[int]=None,

                        fermi:float=None,
                        fermi_shift:float=0,
                        fermi_tolerance:float=0.1,
                        
                        spin_texture: bool=False,
                        calculate_fermi_speed: bool=False,
                        calculate_fermi_velocity: bool=False,
                        calculate_effective_mass: bool=False,

                        supercell:List[int]=[1, 1, 1],
                        extended_zone_directions:List[List[int] or Tuple[int,int,int]]=None,
                        interpolation_factor:int=1,
                        projection_accuracy:str="normal",

                        # plotting saving options
                        plot_brillouin_zone:bool=True,
                        arrow_color: List[str] or List[Tuple[float,float,float]]=None,
                        arrow_size: float=0.1,
                        spin_colors: List[str] or List[Tuple[float,float,float]]=None,
                        colors: List[str] or List[Tuple[float,float,float]]=None,
                        cmap:str="jet",
                        vmin:float=0,
                        vmax:float=1,

                        # saving options
                        show:bool=True,
                        plot_directional_arrows:bool=True,
                        camera_pos:List[float]=[1, 1, 1],
                        background_color:str or Tuple[float,float,float,float]="white",
                        perspective:bool=True,
                        save_2d:bool=None,
                        save_gif:str=None,
                        save_mp4:str=None,
                        save_3d:str=None
        ):
        ################################################################
        # Initialize the Fermi Surface 
        ################################################################
        if fermi is not None:
            self.e_fermi = None

        if (mode == 'property_projection' and 
            calculate_effective_mass== False and 
            calculate_fermi_speed==False and 
            calculate_fermi_velocity==False): 
            raise Exception("Turn one property (calculate_fermi_speed,calculate_fermi_velocity,calculate_effective_mass) to True")

       
        spd, spd_spin, bands_to_keep,spins = self.__format_data(
                                                        mode=mode,
                                                        bands=bands,
                                                        atoms=atoms,
                                                        orbitals=orbitals,
                                                        spins=spins,
                                                        spin_texture=spin_texture,
                                                        fermi_tolerance=fermi_tolerance)
        fermi_surfaces = []
        for ispin, spin in enumerate(spins):
            if spin_colors:
                surface_color = spin_colors[ispin]
            else:
                surface_color = None
            fermi_surface3D = FermiSurface3D(
                                            kpoints=self.parser.ebs.kpoints,
                                            bands=self.parser.ebs.bands[:,:,spin],
                                            bands_to_keep = bands_to_keep,
                                            spd=spd[:,:,ispin],
                                            spd_spin=spd_spin,
                                            colors = colors,
                                            surface_color=surface_color,
                                            fermi=self.e_fermi,
                                            fermi_shift = fermi_shift,
                                            fermi_tolerance=fermi_tolerance,
                                            reciprocal_lattice=self.reciprocal_lattice,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy=projection_accuracy,
                                            supercell=supercell,
                                            cmap=cmap,
                                            vmin=vmin,
                                            vmax=vmax,
                                        )

            if calculate_fermi_speed:
                fermi_surface3D.calculate_fermi_speed()
            elif calculate_fermi_velocity:
                fermi_surface3D.calculate_fermi_velocity()
            elif calculate_effective_mass:
                fermi_surface3D.calculate_effective_mass()

            if mode =='parametric':
                fermi_surface3D.project_atomic_projections()

            if mode =='spin_texture':
                fermi_surface3D.project_spin_texture_atomic_projections()

            if extended_zone_directions:
                fermi_surface3D.extend_surface(extended_zone_directions=extended_zone_directions)
            fermi_surfaces.append(fermi_surface3D)

        fermi_surface=None
        for i,surface in enumerate(fermi_surfaces):
            if i == 0:
                fermi_surface=surface
            else:
                fermi_surface+=surface


        


        ################################################################
        # Initialize the Plotter
        ################################################################

        plotter = pv.Plotter()
        if plot_brillouin_zone:
            plotter.add_mesh(
                fermi_surface3D.brillouin_zone,
                style="wireframe",
                line_width=3.5,
                color="black",
            )


        # Collecting plot options given the mode and other parameters
        options_dict = self.__plot_options_helper(mode=mode,
                                                calculate_fermi_speed=calculate_fermi_speed,
                                                calculate_fermi_velocity=calculate_fermi_velocity,
                                                calculate_effective_mass=calculate_effective_mass)

        # Adding meshes
        if options_dict['scalars'] ==  "spin_magnitude" or options_dict['scalars'] ==  "Fermi Velocity Vector_magnitude":
            arrows = fermi_surface3D.glyph(orient=options_dict['vector_name'],scale=False ,factor=arrow_size)
            if arrow_color is None:
                plotter.add_mesh(arrows, cmap=cmap, show_scalar_bar=False)
            else:
                plotter.add_mesh(arrows, color=arrow_color,show_scalar_bar=False)

        plotter.add_mesh(fermi_surface,
                        scalars =  options_dict['scalars'], 
                        cmap=cmap,
                        show_scalar_bar=False,
                        rgba=options_dict['use_rgba'])

        # Add in custom scalar bar
        if mode != "plain" or spin_texture:
            plotter.add_scalar_bar(
                title=options_dict['text'],
                n_labels=6,
                italic=False,
                bold=False,
                title_font_size=None,
                label_font_size=None,
                position_x=0.4,
                position_y=0.01,
                color="black",)


        # Other plotting options
        if plot_directional_arrows:
            plotter.add_axes(
                xlabel="Kx", 
                ylabel="Ky", 
                zlabel="Kz", 
                line_width=6,
                color='black',
            labels_off=False)

        if not perspective:
            plotter.enable_parallel_projection()
        plotter.set_background(background_color)

        # save and showing setting
        if show:
            plotter.show(cpos=camera_pos, screenshot=save_2d)
        if save_gif is not None:
            path = plotter.generate_orbital_path(n_points=36)
            plotter.open_gif(save_gif)
            plotter.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
        if save_mp4:
            path = plotter.generate_orbital_path(n_points=36)
            plotter.open_movie(save_mp4)
            plotter.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)

        if save_3d:
            plotter.save_meshio(save_3d,  fermi_surface3D)
        plotter.close()

        return None

    def plot_fermi_isoslider(self, 
                        mode:str,
                        iso_range: float,
                        iso_surfaces: int,
                        bands:List[int]=None, 
                        atoms:List[int]=None,
                        orbitals:List[int]=None,
                        spins:List[int]=None,

                        fermi:float=None,
                        fermi_shift:float=0,
                        fermi_tolerance:float=0.1,
                        
                        spin_texture: bool=False,
                        calculate_fermi_speed: bool=False,
                        calculate_fermi_velocity: bool=False,
                        calculate_effective_mass: bool=False,

                        supercell:List[int]=[1, 1, 1],
                        extended_zone_directions:List[List[int]]=None,
                        interpolation_factor:int=1,
                        projection_accuracy:str="normal",

                        # plotting saving options
                        plot_brillouin_zone:bool=True,
                        arrow_color: List[str] or List[Tuple[float,float,float]]=None,
                        arrow_size: float=0.015,
                        colors: List[str] or List[Tuple[float,float,float]]=None,
                        spin_colors: List[str] or List[Tuple[float,float,float]]=None,
                        cmap:str="jet",
                        vmin:float=0,
                        vmax:float=1,

                        # saving options
                        show:bool=True,
                        plot_directional_arrows:bool=True,
                        camera_pos:List[float]=[1, 1, 1],
                        background_color:str or Tuple[float,float,float,float]="white",
                        perspective:bool=True,
                        save_2d:bool=None,
        ):
        ################################################################
        # callback function for the isoslider
        ################################################################
        def create_mesh(plotter:pv.Plotter,
                        value:float,
                        ):
            res = int(value)
            closest_idx = find_nearest(energy_values, res)
            options_dict = self.__plot_options_helper(mode=mode,
                                                calculate_fermi_speed=calculate_fermi_speed,
                                                calculate_fermi_velocity=calculate_fermi_velocity,
                                                calculate_effective_mass=calculate_effective_mass)
            
                
            plotter.add_mesh(e_surfaces[closest_idx],
                            scalars =  options_dict['scalars'], 
                            name='iso_surface', 
                            cmap=cmap,
                            show_scalar_bar=False,
                            rgba=options_dict['use_rgba'])


            if mode != "plain" or spin_texture:
                plotter.add_scalar_bar(
                    title=options_dict['text'],
                    n_labels=6,
                    italic=False,
                    bold=False,
                    title_font_size=None,
                    label_font_size=None,
                    position_x=0.4,
                    position_y=0.01,
                    color="black",)
            
            if options_dict['scalars'] == "spin_magnitude" or options_dict['scalars'] == "Fermi Velocity Vector_magnitude":
                if arrow_color is None:
                    
                    arrows=e_surfaces[closest_idx].glyph(
                                                        orient=options_dict['vector_name'], 
                                                        scale=False ,
                                                        factor=arrow_size)

                    # To update arrows. First ininitialize actor(There will already be a iso_surface actor), then remove, then add
                    arrow_actor = [value for key, value in plotter.renderer.actors.items() if 'PolyData' in key]
                    if len(arrow_actor) != 0:
                        plotter.remove_actor(arrow_actor[0])
                    plotter.add_mesh(arrows, 
                                    cmap=cmap,
                                    show_scalar_bar=False)
                else:
                    plotter.add_mesh(arrows, color=arrow_color,show_scalar_bar=False)

            return None


        ################################################################
        # Generating the isosurafces
        ################################################################
        if fermi is not None:
            self.e_fermi = None

        if mode == 'property_projection' and calculate_effective_mass== False and calculate_fermi_speed==False and calculate_fermi_velocity==False: 
            raise Exception("Turn one property (calculate_fermi_speed,calculate_fermi_velocity,calculate_effective_mass) to True")

        spd, spd_spin, bands_to_keep,spins = self.__format_data(
                                                        mode=mode,
                                                        bands=bands,
                                                        atoms=atoms,
                                                        orbitals=orbitals,
                                                        spins=spins,
                                                        spin_texture=spin_texture,
                                                        fermi_tolerance=fermi_tolerance)

        energy_values = np.linspace(self.e_fermi-iso_range/2,self.e_fermi+iso_range/2,iso_surfaces)
        e_surfaces = []
        for e_value in energy_values:
            fermi_surfaces = []
            for ispin, spin in enumerate(spins):
                if spin_colors:
                    surface_color = spin_colors[ispin]
                else:
                    surface_color = None
                fermi_surface3D = FermiSurface3D(
                                            kpoints=self.parser.ebs.kpoints,
                                            bands=self.parser.ebs.bands[:,:,spin],
                                            bands_to_keep = bands_to_keep,
                                            spd=spd[:,:,ispin],
                                            spd_spin=spd_spin,
                                            colors = colors,
                                            surface_color=surface_color,
                                            fermi=e_value,
                                            fermi_shift = fermi_shift,
                                            fermi_tolerance=fermi_tolerance,
                                            reciprocal_lattice=self.reciprocal_lattice,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy=projection_accuracy,
                                            supercell=supercell,
                                            cmap=cmap,
                                            vmin = vmin,
                                            vmax=vmax,
                                        )
                if calculate_fermi_speed:
                    fermi_surface3D.calculate_fermi_speed()
                elif calculate_fermi_velocity:
                    fermi_surface3D.calculate_fermi_velocity()
                elif calculate_effective_mass:
                    fermi_surface3D.calculate_effective_mass()

                if mode =='parametric':
                    fermi_surface3D.project_atomic_projections()

                if mode =='spin_texture':
                    fermi_surface3D.project_spin_texture_atomic_projections()

                if extended_zone_directions:
                    fermi_surface3D.extend_surface(extended_zone_directions=extended_zone_directions)

                brillouin_zone = fermi_surface3D.brillouin_zone
                fermi_surfaces.append(fermi_surface3D)

            total_surface =None
            for i, surface in enumerate(fermi_surfaces):
                if i == 0:
                    total_surface  = surface
                else:
                    total_surface += surface

            e_surfaces.append(total_surface)


        ################################################################
        # Initialize the Plotter
        ################################################################
        plotter = pv.Plotter()
        if plot_brillouin_zone:
            plotter.add_mesh(
                fermi_surface3D.brillouin_zone,
                style="wireframe",
                line_width=3.5,
                color="black",
            )
        
        custom_callback = functools.partial(create_mesh, plotter)
        plotter.add_slider_widget(custom_callback, [np.amin(energy_values), np.amax(energy_values)], 
                                title='Energy iso-value',
                                style='modern',
                                color = 'black')
        if plot_directional_arrows:
            plotter.add_axes(
                xlabel="Kx", 
                ylabel="Ky", 
                zlabel="Kz",
                color='black',
                line_width=6, 
            labels_off=False)

        if not perspective:
            plotter.enable_parallel_projection()

        plotter.set_background(background_color)

        if show:
            plotter.show(cpos=camera_pos, screenshot=save_2d)

        return None

    def create_isovalue_gif(self, 
                        mode:str,
                        iso_range: float=3,
                        iso_surfaces: int=10,
                        iso_values: List[float]=None,
                        bands:List[int]=None, 
                        atoms:List[int]=None,
                        orbitals:List[int]=None,
                        spins:List[int]=None,

                        fermi:float=None,
                        fermi_shift:float=0,
                        fermi_tolerance:float=0.1,
                        
                        spin_texture: bool=False,
                        calculate_fermi_speed: bool=False,
                        calculate_fermi_velocity: bool=False,
                        calculate_effective_mass: bool=False,

                        supercell:List[int]=[1, 1, 1],
                        extended_zone_directions:List[List[int]]=None,
                        interpolation_factor:int=1,
                        projection_accuracy:str="normal",

                        # plotting saving options
                        plot_brillouin_zone:bool=True,
                        arrow_color: List[str] or List[Tuple[float,float,float]]=None,
                        arrow_size: float=0.015,
                        colors: List[str] or List[Tuple[float,float,float]]=None,
                        spin_colors: List[str] or List[Tuple[float,float,float]]=None,
                        cmap:str="jet",
                        vmin:float=0,
                        vmax:float=1,

                        # saving options
                        plot_directional_arrows:bool=True,
                        camera_pos:List[float]=[1, 1, 1],
                        background_color:str or Tuple[float,float,float,float]="white",
                        save_gif:str=None,

        ):
        if fermi is not None:
            self.e_fermi = None

        if mode == 'property_projection' and calculate_effective_mass== False and calculate_fermi_speed==False and calculate_fermi_velocity==False: 
            raise Exception("Turn one property (calculate_fermi_speed,calculate_fermi_velocity,calculate_effective_mass) to True")

        spd, spd_spin, bands_to_keep,spins = self.__format_data(
                                                        mode=mode,
                                                        bands=bands,
                                                        atoms=atoms,
                                                        orbitals=orbitals,
                                                        spins=spins,
                                                        spin_texture=spin_texture,
                                                        fermi_tolerance=fermi_tolerance)
        
        energy_values = np.linspace(self.e_fermi-iso_range/2,self.e_fermi+iso_range/2,iso_surfaces)
        if iso_values:
            energy_values=iso_values
        e_surfaces = []

        
        for e_value in energy_values:
            fermi_surfaces = []
            for ispin, spin in enumerate(spins):
                if spin_colors:
                    surface_color = spin_colors[ispin]
                else:
                    surface_color = None
                fermi_surface3D = FermiSurface3D(
                                            kpoints=self.parser.ebs.kpoints,
                                            bands=self.parser.ebs.bands[:,:,spin],
                                            bands_to_keep = bands_to_keep,
                                            spd=spd[:,:,ispin],
                                            spd_spin=spd_spin,
                                            colors = colors,
                                            surface_color=surface_color,
                                            fermi=e_value,
                                            fermi_shift = fermi_shift,
                                            fermi_tolerance=fermi_tolerance,
                                            reciprocal_lattice=self.reciprocal_lattice,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy=projection_accuracy,
                                            supercell=supercell,
                                            cmap=cmap,
                                            vmin = vmin,
                                            vmax=vmax,
                                        )
                if calculate_fermi_speed:
                    fermi_surface3D.calculate_fermi_speed()
                elif calculate_fermi_velocity:
                    fermi_surface3D.calculate_fermi_velocity()
                elif calculate_effective_mass:
                    fermi_surface3D.calculate_effective_mass()
                if mode =='parametric':
                    fermi_surface3D.project_atomic_projections()
                if mode =='spin_texture':
                    fermi_surface3D.project_spin_texture_atomic_projections()
                if extended_zone_directions:
                    fermi_surface3D.extend_surface(extended_zone_directions=extended_zone_directions)

                brillouin_zone = fermi_surface3D.brillouin_zone
                fermi_surfaces.append(fermi_surface3D)

            total_surface =None
            for i, surface in enumerate(fermi_surfaces):
                if i == 0:
                    total_surface  = surface
                else:
                    total_surface += surface
            e_surfaces.append(total_surface)

        plotter = pv.Plotter(off_screen=True)
        plotter.open_gif(save_gif)

        if plot_brillouin_zone:
            plotter.add_mesh(
                fermi_surface3D.brillouin_zone,
                style="wireframe",
                line_width=3.5,
                color="black",
            )

        # Initial Mesh
        surface = copy.deepcopy(e_surfaces[0])
        initial_e_value= energy_values[0]

        # Collecting plot options given the mode and other parameters
        options_dict = self.__plot_options_helper(mode=mode,
                                                calculate_fermi_speed=calculate_fermi_speed,
                                                calculate_fermi_velocity=calculate_fermi_velocity,
                                                calculate_effective_mass=calculate_effective_mass)
 
        # Adding meshes
        if options_dict['scalars'] ==  "spin_magnitude" or options_dict['scalars'] ==  "Fermi Velocity Vector_magnitude":
            arrows = fermi_surface3D.glyph(orient=options_dict['vector_name'],scale=False ,factor=arrow_size)
            if arrow_color is None:
                plotter.add_mesh(arrows, cmap=cmap, show_scalar_bar=False)
            else:
                plotter.add_mesh(arrows, color=arrow_color,show_scalar_bar=False)

        plotter.add_mesh(surface,
                scalars =  options_dict['scalars'], 
                cmap=cmap,
                show_scalar_bar=False,
                rgba=options_dict['use_rgba'])

        if mode != "plain" or spin_texture:
            plotter.add_scalar_bar(
                title=options_dict['text'],
                n_labels=6,
                italic=False,
                bold=False,
                title_font_size=None,
                label_font_size=None,
                position_x=0.4,
                position_y=0.01,
                color="black",)

        plotter.add_text(f'Energy Value : {initial_e_value:.4f} eV', color = 'black')

        if plot_directional_arrows:
            plotter.add_axes(
                xlabel="Kx", 
                ylabel="Ky", 
                zlabel="Kz",
                color='black',
                line_width=6, 
            labels_off=False)

        plotter.show(auto_close=False)

        # Run through each frame
        for e_surface,evalues in zip(e_surfaces,energy_values):
            surface.overwrite(e_surface)
            if options_dict['scalars'] ==  "spin_magnitude" or options_dict['scalars'] ==  "Fermi Velocity Vector_magnitude":
                e_arrows = e_surface.glyph(orient=options_dict['vector_name'],scale=False ,factor=arrow_size)
                arrows.overwrite(e_arrows)
            text = f'Energy Value : {evalues:.4f} eV'
            plotter.textActor.SetText(2, text)

            plotter.write_frame()
        # Run backward through each frame
        for e_surface, evalues in zip(e_surfaces[::-1],energy_values[::-1]):
            surface.overwrite(e_surface)
            if options_dict['scalars'] ==  "spin_magnitude" or options_dict['scalars'] ==  "Fermi Velocity Vector_magnitude":
                e_arrows = e_surface.glyph(orient=options_dict['vector_name'],scale=False ,factor=arrow_size)
                arrows.overwrite(e_arrows)
            plotter.write_frame()
            text = f'Energy Value : {evalues:.4f} eV'
            plotter.textActor.SetText(2, text)

        plotter.close()

    def plot_fermi_cross_section(self,
                                mode:str,
                                show_cross_section_area:bool=False,
                                slice_normal: Tuple[float,float,float]=(1,0,0),
                                slice_origin: Tuple[float,float,float]=(0,0,0),
                                line_width:float=5.0,
                                bands:List[int]=None, 
                                atoms:List[int]=None,
                                orbitals:List[int]=None,
                                spins:List[int]=None,

                                fermi:float=None,
                                fermi_shift:float=0,
                                fermi_tolerance:float=0.1,
                                
                                spin_texture: bool=False,
                                calculate_fermi_speed: bool=False,
                                calculate_fermi_velocity: bool=False,
                                calculate_effective_mass: bool=False,

                                supercell:List[int]=[1, 1, 1],
                                extended_zone_directions:List[List[int]]=None,
                                interpolation_factor:int=1,
                                projection_accuracy:str="normal",

                                # plotting saving options
                                plot_brillouin_zone:bool=True,
                                arrow_color: str or List[Tuple[float,float,float]]=None,
                                arrow_size: float=0.015,
                                colors: List[str] or List[Tuple[float,float,float]]=None,
                                spin_colors: List[str] or List[Tuple[float,float,float]]=None,
                                cmap:str="jet",
                                vmin:float=0,
                                vmax:float=1,

                                # saving options
                                show:bool=True,
                                camera_pos:List[float]=[1, 1, 1],
                                plot_directional_arrows:bool=True,
                                background_color:str or Tuple[float,float,float,float]="white",
                                perspective:bool=True,
                                save_2d:bool=None,
                                save_2d_slice:str=None,
                                save_gif:str=None,
                                save_mp4:str=None,
                                save_3d:str=None
        ):

        ################################################################
        # Initialize the Fermi Surface 
        ################################################################
        if fermi is not None:
            self.e_fermi = None

        if mode == 'property_projection' and calculate_effective_mass== False and calculate_fermi_speed==False and calculate_fermi_velocity==False: 
            raise Exception("Turn one property (calculate_fermi_speed,calculate_fermi_velocity,calculate_effective_mass) to True")

        spd, spd_spin, bands_to_keep, spins = self.__format_data(
                                                        mode=mode,
                                                        bands=bands,
                                                        atoms=atoms,
                                                        orbitals=orbitals,
                                                        spins=spins,
                                                        spin_texture=spin_texture,
                                                        fermi_tolerance=fermi_tolerance)
        fermi_surfaces = []
        for ispin, spin in enumerate(spins):
            if spin_colors:
                surface_color = spin_colors[ispin]
            else:
                surface_color = None
            fermi_surface3D = FermiSurface3D(
                                            kpoints=self.parser.ebs.kpoints,
                                            bands=self.parser.ebs.bands[:,:,spin],
                                            bands_to_keep = bands_to_keep,
                                            spd=spd[:,:,ispin],
                                            spd_spin=spd_spin,
                                            colors = colors,
                                            surface_color=surface_color,
                                            fermi=self.e_fermi,
                                            fermi_shift = fermi_shift,
                                            fermi_tolerance=fermi_tolerance,
                                            reciprocal_lattice=self.reciprocal_lattice,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy=projection_accuracy,
                                            supercell=supercell,
                                            cmap=cmap,
                                            vmin = vmin,
                                            vmax=vmax,
                                        )
            if calculate_fermi_speed:
                fermi_surface3D.calculate_fermi_speed()
            elif calculate_fermi_velocity:
                fermi_surface3D.calculate_fermi_velocity()
            elif calculate_effective_mass:
                fermi_surface3D.calculate_effective_mass()
            if mode =='parametric':
                fermi_surface3D.project_atomic_projections()
            if mode =='spin_texture':
                fermi_surface3D.project_spin_texture_atomic_projections()

            if extended_zone_directions:
                fermi_surface3D.extend_surface(extended_zone_directions=extended_zone_directions)
            fermi_surfaces.append(fermi_surface3D)

        fermi_surface=None
        for i,surface in enumerate(fermi_surfaces):
            if i == 0:
                fermi_surface=surface
            else:
                fermi_surface+=surface

        ################################################################
        # Initialize the plotter
        ################################################################
        
        plotter = pv.Plotter()
        if plot_brillouin_zone:
            plotter.add_mesh(
                fermi_surface3D.brillouin_zone,
                style="wireframe",
                line_width=3.5,
                color="black",
            )

        # Collecting plot options given the mode and other parameters
        options_dict = self.__plot_options_helper(mode=mode,
                                                calculate_fermi_speed=calculate_fermi_speed,
                                                calculate_fermi_velocity=calculate_fermi_velocity,
                                                calculate_effective_mass=calculate_effective_mass)

        # if options_dict['scalars'] ==  "spin_magnitude" or options_dict['scalars'] ==  "Fermi Velocity Vector_magnitude":

        #     arrows = fermi_surface.glyph(orient=options_dict['vector_name'],scale=False ,factor=arrow_size)
        #     tot_surface = fermi_surface + arrows
        #     # options_dict['scalars'] = 'spin_magnitude'
        # else:
        #     tot_surface= fermi_surface
        # if arrow_color:
        #     options_dict['scalars']= None
        tot_surface= fermi_surface
        add_custom_mesh_slice(plotter = plotter, 
                                mesh=tot_surface,
                                options_dict=options_dict, 
                                show_cross_section_area=show_cross_section_area,
                                line_width=line_width,
                                normal=slice_normal, 
                                origin = slice_origin,
                                cmap=cmap,
                                arrow_color=arrow_color,
                                scalars = options_dict['scalars'])

        if mode != "plain" or spin_texture:
            plotter.add_scalar_bar(
                title=options_dict['text'],
                n_labels=6,
                italic=False,
                bold=False,
                title_font_size=None,
                label_font_size=None,
                position_x=0.4,
                position_y=0.01,
                color="black",)

        if plot_directional_arrows:
            plotter.add_axes(
                xlabel="Kx", 
                ylabel="Ky", 
                zlabel="Kz", 
                line_width=6,
                color='black', 
            labels_off=False)

        if not perspective:
            plotter.enable_parallel_projection()
        plotter.set_background(background_color)
        if show:
            plotter.show(cpos=camera_pos, screenshot=save_2d)
        if save_2d_slice:
            slice_2d = plotter.plane_sliced_meshes[0]
            plotter.close()
            point1 = slice_2d.points[0,:]
            point2 = slice_2d.points[1,:]
            normal_vec = np.cross(point1,point2)
            p = pv.Plotter()

            if options_dict['vector_name']:
                arrows = slice_2d.glyph(orient=options_dict['vector_name'], scale=False, factor=0.1)
            if arrow_color is not None:
                p.add_mesh(arrows, color=arrow_color, show_scalar_bar=False,name='arrows')
            else:
                p.add_mesh(arrows, cmap=cmap, show_scalar_bar=False,name='arrows')
            p.add_mesh(slice_2d,line_width=line_width)
            p.remove_scalar_bar()
            # p.set_background(background_color)
            p.view_vector(normal_vec)
            p.show(screenshot=save_2d_slice,interactive=False)

    def plot_fermi_surface_area_vs_isovalue(self,
                        iso_range: float=3,
                        iso_surfaces: int=10,
                        iso_values: List[float]=None,
                        spins: List[int] = None,
                        fermi:float=None,
                        fermi_shift:float=0,
                        fermi_tolerance:float=0.1,
                        supercell:List[int]=[1, 1, 1],
                        extended_zone_directions:List[List[int]]=None,
                        interpolation_factor:int=1,
                        # plotting saving options
                        cmap:str="jet",
                        # saving options
                        show:bool=True,
                        savefig:str=None
        ):

        if fermi is not None:
            self.e_fermi = None


        spd, spd_spin, bands_to_keep,spins = self.__format_data(
                                                        mode='plain',
                                                        bands=None,
                                                        atoms=None,
                                                        orbitals=None,
                                                        spins=None,
                                                        spin_texture=False,
                                                        fermi_tolerance=fermi_tolerance)
        
        energy_values = np.linspace(self.e_fermi-iso_range/2,self.e_fermi+iso_range/2,iso_surfaces)
        if iso_values:
            energy_values=iso_values
        e_surfaces = []
        surface_areas = []

        
        for e_value in energy_values:
            fermi_surfaces = []
            for ispin, spin in enumerate(spins):
                fermi_surface3D = FermiSurface3D(
                                            kpoints=self.parser.ebs.kpoints,
                                            bands=self.parser.ebs.bands[:,:,spin],
                                            bands_to_keep = bands_to_keep,
                                            spd=spd[:,:,:],
                                            spd_spin=spd_spin,
                                            colors = None,
                                            fermi=e_value,
                                            fermi_shift = fermi_shift,
                                            fermi_tolerance=fermi_tolerance,
                                            reciprocal_lattice=self.reciprocal_lattice,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy="normal",
                                            supercell=supercell,
                                            cmap=cmap,
                                            vmin=0,
                                            vmax=1,
                                        )

                brillouin_zone = fermi_surface3D.brillouin_zone
                fermi_surfaces.append(fermi_surface3D)

            fermi_surface=None
            for i,surface in enumerate(fermi_surfaces):
                if i == 0:
                    fermi_surface=surface
                else:
                    fermi_surface+=surface
                
            surface_areas.append(fermi_surface.fermi_surface_area)
            e_surfaces.append(fermi_surface)

        fig, axs = plt.subplots(1,1)

        axs.plot(energy_values, surface_areas)

        axs.axvline(self.e_fermi, color='blue', linestyle="dotted", linewidth=1)
        axs.set_xlabel("Energy (eV)")
        axs.set_ylabel("Surface Area (m$^{-2}$)")

        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)

    def __parse_code(self):

        """_summary_
        Helper method to handle the parsing of the codes
        Returns:
            _type_: _description_
        """
        if self.code == "vasp":
            if self.dirname is None:
                self.dirname = "fermi"
            outcar_file = f"{self.dirname}{os.sep}OUTCAR"
            poscar_file = f"{self.dirname}{os.sep}POSCAR"
            procar_file = f"{self.dirname}{os.sep}PROCAR"

            if self.repair:
                repairhandle = UtilsProcar()
                repairhandle.ProcarRepair(procar_file, procar_file)
                print("PROCAR repaired. Run with repair=False next time.")

            outcar = io.vasp.Outcar(filename=outcar_file)
            e_fermi = outcar.efermi
            
            poscar = io.vasp.Poscar(filename=poscar_file)
            structure = poscar.structure
            reciprocal_lattice = poscar.structure.reciprocal_lattice

            parser = io.vasp.Procar(filename=procar_file,
                                    structure=structure,
                                    reciprocal_lattice=reciprocal_lattice,
                                    efermi=e_fermi,
                                    )
            if self.apply_symmetry:
                parser.ebs.ibz2fbz(outcar.rotations)
            # data = ProcarSelect(procarFile, deepCopy=True)

        elif self.code == "qe":
            if self.dirname is None:
                self.dirname = "fermi"
            parser = io.qe.QEParser( dirname = self.dirname, scf_in_filename = "scf.in", bands_in_filename = "bands.in", 
                                pdos_in_filename = "pdos.in", kpdos_in_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml")
            reciprocal_lattice = parser.reciprocal_lattice

            e_fermi = parser.efermi
            # e_fermi = 0
            if self.apply_symmetry:
                parser.ebs.ibz2fbz(parser.rotations)

        elif self.code == "lobster":
            procarFile = LobsterFermiParser()
            reciprocal_lattice = procarFile.reclat
            parser = ProcarSelect(procarFile, deepCopy=True)
            e_fermi = 0

        elif self.code == "bxsf":
            if self.dirname is None:
                self.dirname = "fermi"

            infiles = [f"{self.dirname}{os.sep}{file}" for file in os.listdir(self.dirname) if 'bxsf' in file]
          
            parser = io.bxsf.BxsfParser(infiles=infiles)

            e_fermi = parser.e_fermi
            # e_fermi = 0
            reciprocal_lattice = parser.reciprocal_lattice
            procarFile = None

        elif self.code == "frmsf":
            if self.dirname is None:
                self.dirname = "fermi"
            infile = f"{self.dirname}{os.sep}in.frmsrf"

            e_fermi = 0

            parser = io.frmsf.FrmsfParser(infile=infile)
            reciprocal_lattice = parser.rec_lattice
            bands = np.arange(len(parser.bands[0, :]))
            procarFile = None


        elif self.code == "abinit":
            if self.dirname is None:
                self.dirname = "fermi"
            outfile = f"{self.dirname}{os.sep}abinit.out"

            # e_fermi = 0

            output = io.abinit.Output(abinit_output=outfile)
            # e_fermi = 0
            e_fermi = output.fermi
            
            # poscar = io.vasp.Poscar(filename=poscar_file)
            structure = output.structure
            reciprocal_lattice = output.structure.reciprocal_lattice

            parser = io.abinit.Procar(
                                filename=self.dirname,
                                abinit_output=outfile,
                                structure=output.structure,
                                reciprocal_lattice=output.reclat,
                                kpath=None,
                                efermi=output.fermi,
                            )

            parser.ebs.bands += e_fermi
        parser.ebs.bands += e_fermi
        return parser, reciprocal_lattice, e_fermi

    def __format_data(self, 
                    mode:str,
                    bands:List[int]=None,
                    atoms:List[int]=None,
                    orbitals:List[int]=None,
                    spins:List[int]=None, 
                    spin_texture: bool=False,
                    fermi_tolerance:float=0.1,):

        bands_to_keep = bands
        if bands_to_keep is None:
            bands_to_keep = np.arange(len(self.parser.ebs.bands[0, :,0]))

        self.band_near_fermi = []
        for iband in range(len(self.parser.ebs.bands[0,:,0])):
            fermi_surface_test = len(np.where(np.logical_and(self.parser.ebs.bands[:,iband,0]>=self.e_fermi-fermi_tolerance, self.parser.ebs.bands[:,iband,0]<=self.e_fermi+fermi_tolerance))[0])
            if fermi_surface_test != 0:
                self.band_near_fermi.append(iband)

        print(f"Fermi Energy : {self.e_fermi}")
        print(f"Bands near the fermi energy : {self.band_near_fermi}")

        if spins is None:
            if self.parser.ebs.bands.shape[2] == 1 or np.all(self.parser.ebs.bands[:,:,1]==0):
                spins = [0]
            else:
                spins = [0,1]
        
        spd = []
        if mode == "parametric":
            if orbitals is None and self.parser.ebs.projected is not None:
                orbitals = np.arange(self.parser.ebs.norbitals, dtype=int)
            if atoms is None and self.parser.ebs.projected is not None:
                atoms = np.arange(self.parser.ebs.natoms, dtype=int)
            projected = self.parser.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
            # projected = projected[:,:,spins[0]]
            for ispin in spins:
                spin_bands_projections = []
                for iband in bands_to_keep:
                    spin_bands_projections.append(projected[:,iband,ispin])
                spd.append( spin_bands_projections)
            spd = np.array(spd).T
            
        else:
            spd = np.zeros(shape = (self.parser.ebs.nkpoints,len(bands_to_keep),len(spins)))

    
        spd_spin = []

        if spin_texture:
            ebsX = copy.deepcopy(self.parser.ebs)
            ebsY = copy.deepcopy(self.parser.ebs)
            ebsZ = copy.deepcopy(self.parser.ebs)
            print(ebsX.projected.shape)
            ebsX.projected = ebsX.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
            ebsY.projected = ebsY.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
            ebsZ.projected = ebsZ.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

            ebsX.projected = ebsX.projected[:,:,[0]]
            ebsY.projected = ebsY.projected[:,:,[1]]
            ebsZ.projected = ebsZ.projected[:,:,[2]]


            for iband in bands_to_keep:
                spd_spin.append(
                    [ebsX.projected[:, iband], ebsY.projected[:, iband], ebsZ.projected[:, iband]]
                )

            spd_spin = np.array(spd_spin)[:,:,:,0]
            spd_spin = np.swapaxes(spd_spin, 0, 1)
            spd_spin = np.swapaxes(spd_spin, 0, 2)
        else:
            for iband in bands_to_keep:
                spd_spin.append(None)
        
        return spd, spd_spin, bands_to_keep, spins

    def __plot_options_helper(self,
                            mode:str,
                            calculate_fermi_speed:bool=False,
                            calculate_fermi_velocity:bool=False,
                            calculate_effective_mass:bool=False,

                            ):
        
        if mode == "plain":
            text = "plain"
            scalars = "bands"
            vector_name=None
            use_rgba = True

        elif mode == "parametric":
            text = "parametric"
            scalars = "scalars"
            vector_name=None
            use_rgba = False

        elif mode == "property_projection":
            
            use_rgba = False

            if calculate_fermi_speed == True:
                scalars = "Fermi Speed"
                text = "Fermi Speed"
                vector_name=None
            elif calculate_fermi_velocity == True:
                scalars = "Fermi Velocity Vector_magnitude"
                vector_name = "Fermi Velocity Vector"
                text = "Fermi Speed"
            elif calculate_effective_mass == True:
                scalars = "Geometric Average Effective Mass"
                text = "Geometric Average Effective Mass"
                vector_name=None
            else:
                print("Please select a property")
        elif mode == 'spin_texture':
            text = "Spin Texture"
            use_rgba = False
            scalars = "spin_magnitude"
            vector_name = 'spin'

        options_dict = {
                "text":text,
                "scalars":scalars,
                "vector_name":vector_name,
                "use_rgba":use_rgba,
                }


        return options_dict
    # Not used but maybe useful in the future
    def __common_plotting(self,
                        fermi_surface,
                        plotter: pv.Plotter,
                        mode:str, 
                        text:str,
                        spin_texture:bool=False,
                        
                        camera_pos:List[float]=[1, 1, 1],
                        background_color:str or Tuple[float,float,float,float]="white",
                        perspective:bool=True,

                        show:bool=False,
                        save_2d:bool=None,
                        save_gif:str=None,
                        save_mp4:str=None,
                        save_3d:str=None):

        if mode != "plain" or spin_texture:
            plotter.add_scalar_bar(
                title=text,
                n_labels=6,
                italic=False,
                bold=False,
                title_font_size=None,
                label_font_size=None,
                position_x=0.4,
                position_y=0.01,
                color="black",)

        plotter.add_axes(
            xlabel="Kx", 
            ylabel="Ky", 
            zlabel="Kz", 
            line_width=6,
            color='black',
            labels_off=False)

        if not perspective:
            plotter.enable_parallel_projection()

        plotter.set_background(background_color)
        if not show:
            plotter.show(cpos=camera_pos, screenshot=save_2d)
        if save_gif is not None:
            path = plotter.generate_orbital_path(n_points=36)
            plotter.open_gif(save_gif)
            plotter.orbit_on_path(path) 
        if save_mp4:
            path = plotter.generate_orbital_path(n_points=36)
            plotter.open_movie(save_mp4)
            plotter.orbit_on_path(path) 

        if save_3d is not None:
            plotter.save_meshio(save_3d,  fermi_surface)


# Following method is for custom slices
def add_custom_mesh_slice(
                    plotter,
                    mesh, 
                    options_dict:dict,
                    show_cross_section_area:bool=False,
                    line_width:float=5.0,
                    normal='x', 
                    generate_triangles=False,
                    widget_color=None, 
                    assign_to_axis=None,
                    tubing=False, 
                    origin_translation=True, 
                    origin = (0,0,0),
                    outline_translation=False, 
                    implicit=True,
                    normal_rotation=True, 
                    cmap='jet',
                    arrow_color=None,
                    **kwargs):

        name = kwargs.get('name', mesh.memory_address)

        # print(name)
        # if kwargs.get('scalars', mesh.active_scalars_name) != 'spin':
            
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        plotter.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0, line_width=line_width,show_scalar_bar=False, rgba=options_dict['use_rgba'])

        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

    
        plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pv.wrap(alg.GetOutput())
        plotter.plane_sliced_meshes.append(plane_sliced_mesh)
        
        if show_cross_section_area:
            user_slice = plotter.plane_sliced_meshes[0]
            surface = user_slice.delaunay_2d()
            plotter.add_text(f"Cross sectional area : {surface.area:.4f}"+" m^-2", color = 'black')

  
        def callback(normal, origin):
            # create the plane for clipping
            
            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            # plotter.add_mesh(plane_sliced_mesh, name=name+"outline", opacity=0.0, line_width=line_width,show_scalar_bar=False, rgba=options_dict['use_rgba'])
            if options_dict['vector_name']:
                arrows = plane_sliced_mesh.glyph(orient=options_dict['vector_name'], scale=False, factor=0.1)
                
                if arrow_color is not None:
                    plotter.add_mesh(arrows, color=arrow_color, show_scalar_bar=False,name='arrows')
                else:
                    plotter.add_mesh(arrows, cmap=cmap, show_scalar_bar=False,name='arrows')

            
            if show_cross_section_area:
                user_slice = plotter.plane_sliced_meshes[0]
                surface = user_slice.delaunay_2d()
                text = f"Cross sectional area : {surface.area:.4f}"+" m^-2"

                plotter.textActor.SetText(2, text)




        plotter.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal='x',
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=origin,
                              normal_rotation=normal_rotation)

    
        actor = plotter.add_mesh(plane_sliced_mesh,show_scalar_bar=False, line_width=line_width,rgba=options_dict['use_rgba'], **kwargs)
        plotter.plane_widgets[0].SetNormal(normal)
        return actor
    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

