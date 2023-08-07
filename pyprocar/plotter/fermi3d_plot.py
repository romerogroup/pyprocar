import os
import copy
import yaml
from typing import List, Tuple


import numpy as np
import pyvista as pv
from matplotlib import colors as mpcolors
from matplotlib import cm
from PIL import Image

import vtk
from pyvista.core.filters import _get_output  # avoids circular import
from pyvista.core.utilities import (
    NORMALS,
    assert_empty_kwargs,
    generate_plane,
    get_array,
    get_array_association,
    try_callback,
)
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    crinkle_algorithm,
    outline_algorithm,
    pointset_to_polydata_algorithm,
    set_algorithm_input,
)
# from pypro

from pyprocar import io
from pyprocar.core.fermisurface3D import FermiSurface3D
from pyprocar.utils import ROOT


class FermiDataHandler:
    def __init__(self, ebs, fermi_tolerance=0.1):
        self.initial_ebs=copy.copy(ebs)
        self.ebs = ebs
        self.fermi_tolerance = fermi_tolerance
        self.mode=None
        # Other instance variables...

    def process_data(self, mode:str,
                    bands:List[int]=None,
                    atoms:List[int]=None,
                    orbitals:List[int]=None,
                    spins:List[int]=None, 
                    spin_texture: bool=False,
                    fermi_tolerance:float=0.1,):
        """A helper method to process/aggregate data

        Parameters
        ----------
        mode : str
            the mdoe name
        bands : List[int], optional
            List of bands, by default None
        atoms : List[int], optional
            List of stoms, by default None
        orbitals : List[int], optional
            List of orbitals, by default None
        spins : List[int], optional
            List of spins, by default None
        spin_texture : bool, optional
            Boolean to plot spin texture, by default False
        fermi_tolerance : float, optional
            The tolerace to search for bands around the fermi energy, by default 0.1

        Returns
        -------
        _type_
            _description_
        """

        bands_to_keep = bands
        if bands_to_keep is None:
            bands_to_keep = np.arange(len(self.initial_ebs.bands[0, :,0]))

        self.band_near_fermi = []
        for iband in range(len(self.initial_ebs.bands[0,:,0])):
            fermi_surface_test = len(np.where(np.logical_and(self.initial_ebs.bands[:,iband,0]>=self.initial_ebs.efermi-fermi_tolerance, 
                                                             self.initial_ebs.bands[:,iband,0]<=self.initial_ebs.efermi+fermi_tolerance))[0])
            if fermi_surface_test != 0:
                self.band_near_fermi.append(iband)

        
        if spins is None:
            if self.initial_ebs.bands.shape[2] == 1 or np.all(self.initial_ebs.bands[:,:,1]==0):
                spins = [0]
            else:
                spins = [0,1]

        if self.initial_ebs.nspins==2 and spins is None:
            self.spin_pol=[0,1]
        elif self.initial_ebs.nspins==2:
            self.spin_pol=spins
        else:
            self.spin_pol=[0]
        

        spd = []
        if mode == "parametric":
            if orbitals is None and self.initial_ebs.projected is not None:
                orbitals = np.arange(self.initial_ebs.norbitals, dtype=int)
            if atoms is None and self.initial_ebs.projected is not None:
                atoms = np.arange(self.initial_ebs.natoms, dtype=int)

            if self.initial_ebs.is_non_collinear:
                projected = self.initial_ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=True)
            else:
                projected = self.initial_ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

            for ispin in self.spin_pol:
                spin_bands_projections = []
                for iband in bands_to_keep:
                    spin_bands_projections.append(projected[:,iband,ispin])
                spd.append( spin_bands_projections)
            spd = np.array(spd).T
            spins = np.arange(spd.shape[2])
        else:
            spd = np.zeros(shape = (self.initial_ebs.nkpoints,len(bands_to_keep),len(spins)))

        spd_spin = []
        if spin_texture:
            ebs = copy.deepcopy(self.initial_ebs)
            ebs.projected = ebs.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

            for iband in bands_to_keep:
                spd_spin.append(
                    [ebs.projected[:,iband,[1]], ebs.projected[:,iband,[2]], ebs.projected[:,iband,[3]]]
                )
            spd_spin = np.array(spd_spin)[:,:,:,0]
            spd_spin = np.swapaxes(spd_spin, 0, 1)
            spd_spin = np.swapaxes(spd_spin, 0, 2)
        else:
            for iband in bands_to_keep:
                spd_spin.append(None)
        if len(spd_spin)==0:
            spd_spin=None
        self.spd= spd
        self.spd_spin=spd_spin
        self.bands_to_keep=bands_to_keep
        self.spins=spins
        self.mode=mode
        return spd, spd_spin, bands_to_keep, spins

    def get_surface_data(self,
                    property_name=None,
                    fermi:float=None,
                    fermi_shift: float=0.0,
                    interpolation_factor: int=1,
                    projection_accuracy: str="high",
                    supercell: List[int]=[1, 1, 1],
                    extended_zone_directions:List[List[int] or Tuple[int,int,int]]=None,):
        
        if self.mode is None:
            raise "You must call process data function before get_surface"

        if property_name:
            current_emplemented_properties = ['fermi_velocity', 'fermi_speed' ,'harmonic_effective_mass']
            if property_name not in current_emplemented_properties:
                tmp=f"You must choose one of the following properies : {current_emplemented_properties}"
                raise ValueError(tmp)
            else:
                if property_name=='fermi_velocity':
                    self.ebs.bands_gradient
                elif property_name=='fermi_speed':
                    self.ebs.bands_hessian
                elif property_name=='harmonic_effective_mass':
                    self.ebs.bands_hessian
                

        fermi_surfaces = []
        spins_band_index=[]
        spins_index=[]
        for ispin, spin in enumerate(self.spin_pol):
            ebs=copy.copy(self.ebs)
            ebs.bands=ebs.bands[:,self.bands_to_keep,spin]
            fermi_surface3D = FermiSurface3D(
                                            ebs=ebs,
                                            fermi=fermi,
                                            fermi_shift = fermi_shift,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy=projection_accuracy,
                                            supercell=supercell,
                                        )
            self.property_name=property_name
            if self.property_name=='fermi_speed':
                fermi_surface3D.project_fermi_speed(fermi_speed=ebs.fermi_speed[...,fermi_surface3D.band_index_map,ispin])
            elif self.property_name=='fermi_velocity':
                fermi_surface3D.project_fermi_velocity(fermi_velocity=ebs.fermi_velocity[...,fermi_surface3D.band_index_map,ispin])
            elif self.property_name=='harmonic_effective_mass':
                fermi_surface3D.project_harmonic_effective_mass(harmonic_effective_mass=ebs.harmonic_average_effective_mass[...,fermi_surface3D.band_index_map,ispin])

            if self.mode =='parametric':
                fermi_surface3D.project_atomic_projections(self.spd[:,fermi_surface3D.band_index_map,ispin])

            if self.mode =='spin_texture':
                fermi_surface3D.project_spin_texture_atomic_projections(self.spd_spin[:,fermi_surface3D.band_index_map,:])

            if extended_zone_directions:
                fermi_surface3D.extend_surface(extended_zone_directions=extended_zone_directions)
            fermi_surfaces.append(fermi_surface3D)

                
            if ispin==1:
                spin_band_index = list(fermi_surface3D.point_data['band_index'] + len(np.unique(spin_band_index)) )
                spin_band_index.reverse()
            else:
                spin_band_index = list(fermi_surface3D.point_data['band_index'])
                spin_band_index.reverse()
            spins_band_index.extend(spin_band_index)

            spin_index=[ispin]*len(fermi_surface3D.points)
            spin_index.reverse()
            spins_index.extend(spin_index)


        # These reverese are used 
        # because when you combine two meshes the point_data 
        # and points are prepended to each other, so extend would be backwards
        spins_band_index.reverse()
        spins_index.reverse()

        fermi_surface=None
        for i,surface in enumerate(fermi_surfaces):
            if i == 0:
                fermi_surface=surface
            else:
                fermi_surface+=surface

        fermi_surface.point_data['band_index']= np.array(spins_band_index)
        fermi_surface.point_data['spin_index']= np.array(spins_index)
        
        self.fermi_surface=fermi_surface
        return self.fermi_surface
    
class FermiVisualizer:

    def __init__(self, data_handler=None,**kwargs):

        self.data_handler = data_handler
        self.plotter=pv.Plotter()

        with open(os.path.join(ROOT,'pyprocar','cfg','fermi_surface_3d.yml'), 'r') as file:
            self.plotting_options = yaml.safe_load(file)
        self.update_config(kwargs)  
        self._setup_plotter()

    def add_scalar_bar(self,name):
        if self.plotting_options['add_scalar_bar']['value']:
            self.plotter.add_scalar_bar(
                    title=name,
                    n_labels=self.plotting_options['scalar_bar_labels']['value'],
                    italic=self.plotting_options['scalar_bar_italic']['value'],
                    bold=self.plotting_options['scalar_bar_bold']['value'],
                    title_font_size=self.plotting_options['scalar_bar_title_font_size']['value'],
                    label_font_size=self.plotting_options['scalar_bar_label_font_size']['value'],
                    position_x=self.plotting_options['scalar_bar_position_x']['value'],
                    position_y=self.plotting_options['scalar_bar_position_y']['value'],
                    color=self.plotting_options['scalar_bar_color']['value'])
        
    def add_axes(self):
        if self.plotting_options['add_axes']['value']:
            self.plotter.add_axes(
                    xlabel=self.plotting_options['x_axes_label']['value'], 
                    ylabel=self.plotting_options['y_axes_label']['value'], 
                    zlabel=self.plotting_options['z_axes_label']['value'],
                    color=self.plotting_options['axes_label_color']['value'],
                    line_width=self.plotting_options['axes_line_width']['value'],
                labels_off=False)

    def add_brillouin_zone(self,fermi_surface):
       self.plotter.add_mesh(
                fermi_surface.brillouin_zone,
                style=self.plotting_options['brillouin_zone_style']['value'],
                line_width=self.plotting_options['brillouin_zone_line_width']['value'],
                color=self.plotting_options['brillouin_zone_color']['value'],
                opacity=self.plotting_options['brillouin_zone_opacity']['value']
            )
    
    def add_surface(self,surface):

        surface=self._setup_band_colors(surface)
        if self.plotting_options['spin_colors']['value'] != [None,None]:
            spin_colors=[]
            for spin_index in surface.point_data['spin_index']:
                if spin_index == 0:
                    spin_colors.append(self.plotting_options['spin_colors']['value'][0])
                else:
                    spin_colors.append(self.plotting_options['spin_colors']['value'][1])
            surface.point_data['spin_colors']=spin_colors
            self.plotter.add_mesh(surface,
                                scalars='spin_colors',
                                cmap=self.plotting_options['surface_cmap']['value'],
                                clim=self.plotting_options['surface_clim']['value'],
                                show_scalar_bar=False,
                                opacity=self.plotting_options['surface_opacity']['value'],)
                                # rgba=self.data_handler.use_rgba)
            

        elif self.plotting_options['surface_color']['value']:
            self.plotter.add_mesh(surface,
                                color=self.plotting_options['surface_color']['value'],
                                opacity=self.plotting_options['surface_opacity']['value'],)
        else:
            if self.plotting_options['surface_clim']['value']:
                self._normalize_data(surface,scalars_name=self.data_handler.scalars_name)

            self.plotter.add_mesh(surface,
                                scalars=self.data_handler.scalars_name,
                                cmap=self.plotting_options['surface_cmap']['value'],
                                clim=self.plotting_options['surface_clim']['value'],
                                show_scalar_bar=False,
                                opacity=self.plotting_options['surface_opacity']['value'],
                                rgba=self.data_handler.use_rgba)
            
    def add_texture(self,
                    fermi_surface,
                    scalars_name, 
                    vector_name):
        if scalars_name=="spin_magnitude" or scalars_name=="Fermi Velocity Vector_magnitude":
            arrows = fermi_surface.glyph(orient=vector_name,
                                         scale=self.plotting_options['texture_scale']['value'] ,
                                         factor=self.plotting_options['texture_size']['value'])
            if self.plotting_options['texture_color']['value'] is None:
                self.plotter.add_mesh(arrows,scalars=scalars_name, 
                                      cmap=self.plotting_options['texture_cmap']['value'], 
                                      show_scalar_bar=False,
                                      opacity=self.plotting_options['texture_opacity']['value'])
            else:
                self.plotter.add_mesh(arrows,scalars=scalars_name, 
                                      color=self.plotting_options['texture_color']['value'],
                                      show_scalar_bar=False,
                                      opacity=self.plotting_options['texture_opacity']['value'])
        else:
            arrows=None
        return arrows
    
    def add_isoslider(self,e_surfaces, energy_values):
        self.energy_values=energy_values
        self.e_surfaces=e_surfaces
        for i,surface in enumerate(self.e_surfaces):
            self.e_surfaces[i]=self._setup_band_colors(surface)

        self.plotter.add_slider_widget(self._custom_isoslider_callback, 
                                [np.amin(energy_values), np.amax(energy_values)], 
                                title=self.plotting_options['isoslider_title']['value'],
                                style=self.plotting_options['isoslider_style']['value'],
                                color=self.plotting_options['isoslider_color']['value'])
        
        self.add_brillouin_zone(self.e_surfaces[0])
        self.add_axes()
        self.set_background_color()

    def add_isovalue_gif(self,e_surfaces, energy_values,save_gif):
        self.energy_values=energy_values
        self.e_surfaces=e_surfaces
        for i,surface in enumerate(self.e_surfaces):
            self.e_surfaces[i]=self._setup_band_colors(surface)
        
        self.add_brillouin_zone(self.e_surfaces[0])
        self.add_axes()
        self.set_background_color()


        self.plotter.off_screen = True
        self.plotter.open_gif(save_gif)
        # Initial Mesh
        surface = copy.deepcopy(e_surfaces[0])
        initial_e_value= energy_values[0]

        self.plotter.add_text(f'Energy Value : {initial_e_value:.4f} eV', color = 'black')
        arrows = self.add_texture(
                    surface,
                    scalars_name=self.data_handler.scalars_name, 
                    vector_name=self.data_handler.vector_name)
        self.add_surface(surface)
        if self.data_handler.mode != "plain":
            self.add_scalar_bar(name=self.data_handler.scalars_name)

        self.plotter.show(auto_close=False)
        

         # Run through each frame
        for e_surface,evalues in zip(e_surfaces,energy_values):

            surface.copy_from(e_surface)

            # Must set active scalars after calling copy_from
            surface.set_active_scalars(name=self.data_handler.scalars_name)

            text = f'Energy Value : {evalues:.4f} eV'
            self.plotter.textActor.SetText(2, text)

            if self.data_handler.scalars_name=="spin_magnitude" or self.data_handler.scalars_name=="Fermi Velocity Vector_magnitude":
                e_arrows = e_surface.glyph(orient=self.data_handler.vector_name,
                                            scale=self.plotting_options['texture_scale']['value'] ,
                                            factor=self.plotting_options['texture_size']['value'])

                arrows.copy_from(e_arrows)
                arrows.set_active_scalars(name=self.data_handler.scalars_name)
                # arrows.set_active_vectors(name=options_dict['vector_name'])

            self.plotter.write_frame()

        # Run backward through each frame
        for e_surface, evalues in zip(e_surfaces[::-1],energy_values[::-1]):
            surface.copy_from(e_surface)

            # Must set active scalars after calling copy_from
            surface.set_active_scalars(name=self.data_handler.scalars_name)

            text = f'Energy Value : {evalues:.4f} eV'
            self.plotter.textActor.SetText(2, text)

            if self.data_handler.scalars_name=="spin_magnitude" or self.data_handler.scalars_name=="Fermi Velocity Vector_magnitude":
                e_arrows = e_surface.glyph(orient=self.data_handler.vector_name,
                                            scale=self.plotting_options['texture_scale']['value'] ,
                                            factor=self.plotting_options['texture_size']['value'])

                arrows.copy_from(e_arrows)
                arrows.set_active_scalars(name=self.data_handler.scalars_name)
                # arrows.set_active_vectors(name=options_dict['vector_name'])

            self.plotter.write_frame()

        self.plotter.close()
    
    def add_slicer(self,surface,
                                 show=True,
                                 save_2d=None,
                                 save_2d_slice=None,
                                 slice_normal=(1,0,0),
                                 slice_origin=(0,0,0)):
        
        self.add_brillouin_zone(surface)
        self.add_axes()
        self.set_background_color()

        self.add_surface(surface)
        if self.data_handler.mode != "plain":
            self.add_scalar_bar(name=self.data_handler.scalars_name)
        self.add_texture(
                        surface,
                        scalars_name=self.data_handler.scalars_name, 
                        vector_name=self.data_handler.vector_name)
        
        self._add_custom_mesh_slice(mesh=surface,normal=slice_normal,origin=slice_origin)
        
        if show:
            self.plotter.show(cpos=self.plotting_options['plotter_camera_pos']['value'], 
                         screenshot=save_2d)
        if save_2d_slice:
            slice_2d = self.plotter.plane_sliced_meshes[0]
            self.plotter.close()
            point1 = slice_2d.points[0,:]
            point2 = slice_2d.points[1,:]
            normal_vec = np.cross(point1,point2)
            p = pv.Plotter()

            if self.data_handler.vector_name:
                arrows = slice_2d.glyph(orient=self.data_handler.vector_name, scale=False, factor=0.1)
            if self.plotting_options['texture_color']['value'] is not None:
                p.add_mesh(arrows, color=self.plotting_options['texture_color']['value'], show_scalar_bar=False,name='arrows')
            else:
                p.add_mesh(arrows, 
                           cmap=self.plotting_options['texture_cmap']['value'], 
                           show_scalar_bar=False,name='arrows')
            p.add_mesh(slice_2d,
                       line_width=self.plotting_options['cross_section_slice_linewidth']['value'])
            p.remove_scalar_bar()
            # p.set_background(background_color)
            p.view_vector(normal_vec)
            p.show(screenshot=save_2d_slice,interactive=False)

    def add_box_slicer(self,surface,
                                 show=True,
                                 save_2d=None,
                                 save_2d_slice=None,
                                 slice_normal=(1,0,0),
                                 slice_origin=(0,0,0)):
        
        self.add_brillouin_zone(surface)
        self.add_axes()
        self.set_background_color()

        self.add_surface(surface)
        if self.data_handler.mode != "plain":
            self.add_scalar_bar(name=self.data_handler.scalars_name)
        self.add_texture(
                        surface,
                        scalars_name=self.data_handler.scalars_name, 
                        vector_name=self.data_handler.vector_name)
        
        self._add_custom_box_slice_widget(
                    mesh=surface, 
                    show_cross_section_area=self.plotting_options['cross_section_slice_show_area'],
                    normal=slice_normal,
                    origin=slice_origin,
                    )

        if show:
            self.plotter.show(cpos=self.plotting_options['plotter_camera_pos']['value'], 
                         screenshot=save_2d)
        if save_2d_slice:
            slice_2d = self.plotter.plane_sliced_meshes[0]
            self.plotter.close()
            point1 = slice_2d.points[0,:]
            point2 = slice_2d.points[1,:]
            normal_vec = np.cross(point1,point2)
            p = pv.Plotter()

            if self.data_handler.vector_name:
                arrows = slice_2d.glyph(orient=self.data_handler.vector_name, scale=False, factor=0.1)
            if self.plotting_options['texture_color']['value'] is not None:
                p.add_mesh(arrows, color=self.plotting_options['texture_color']['value'], show_scalar_bar=False,name='arrows')
            else:
                p.add_mesh(arrows, 
                           cmap=self.plotting_options['texture_cmap']['value'], 
                           show_scalar_bar=False,name='arrows')
            p.add_mesh(slice_2d,
                       line_width=self.plotting_options['cross_section_slice_linewidth']['value'])
            p.remove_scalar_bar()
            # p.set_background(background_color)
            p.view_vector(normal_vec)
            p.show(screenshot=save_2d_slice,interactive=False)

    def set_background_color(self):
        self.plotter.set_background(self.plotting_options['background_color']['value'])

    def show(self,filename=None):
        if filename:
            file_extentions = filename.split()
            if self.plotting_options['plotter_offscreen']:
                self.plotter.off_screen = True
                self.plotter.show( cpos=self.plotting_options['plotter_camera_pos']['value'],auto_close=False)  
                self.plotter.show(screenshot=filename)
            else:
                image=self.plotter.show( cpos=self.plotting_options['plotter_camera_pos']['value'],screenshot=True)  
                im = Image.fromarray(image)
                im.save(filename)
        else:
            self.plotter.show(cpos=self.plotting_options['plotter_camera_pos']['value'])
        
    def save_gif(self,filename):
        path = self.plotter.generate_orbital_path(n_points=self.plotting_options['orbit_gif_n_points']['value'])
        self.plotter.open_gif(filename)
        self.plotter.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=self.plotting_options['orbit_gif_step']['value'])
    
    def save_mp4(self,filename):
        path = self.plotter.generate_orbital_path(n_points=self.plotting_options['orbit_mp4_n_points']['value'])
        self.plotter.open_movie(filename)
        self.plotter.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=self.plotting_options['orbit_mp4_step']['value'])
    
    def save_mesh(self,filename,surface):
        pv.save_meshio(filename, surface)

    def update_config(self, config_dict):
        for key,value in config_dict.items():
            self.plotting_options[key]['value']=value
    
    def _setup_band_colors(self,fermi_surface):
        band_colors = self.plotting_options['surface_bands_colors']['value']
        if self.plotting_options['surface_bands_colors']['value'] == []:
            band_colors=self._generate_band_colors(fermi_surface)

        fermi_surface = self._apply_fermi_surface_band_colors(fermi_surface,band_colors)
        return fermi_surface

    def _apply_fermi_surface_band_colors(self,fermi_surface,band_colors):
        unique_band_index = np.unique(fermi_surface.point_data['band_index'])
    
        if len(band_colors) != len(unique_band_index):
            raise "You need to list the number of colors as there are bands that make up the surface"
        
        surface_band_colors=[]
        for band_color in band_colors:
            if isinstance(band_color,str):
                surface_color = mpcolors.to_rgba_array(band_color, alpha =1 )[0,:]
                surface_band_colors.append(surface_color)
            else:
                surface_color=band_color
                surface_band_colors.append(surface_color)

        band_colors=[]
        band_surface_indices=fermi_surface.point_data['band_index']
        for band_surface_index in band_surface_indices:
            band_color = surface_band_colors[band_surface_index]
            band_colors.append(band_color)
        fermi_surface.point_data['bands']=band_colors

        return fermi_surface
    
    def _generate_band_colors(self,fermi_surface):
        # Generate unique rgba values for the bands
        unique_band_index = np.unique(fermi_surface.point_data['band_index'])
        nsurface = len(unique_band_index)
        norm = mpcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap(self.plotting_options['surface_cmap']['value'])
        solid_color_surface = np.arange(nsurface ) / nsurface
        band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)
        return band_colors
    
    def _setup_plotter(self):
        """Helper method set parameter values
        """
        if self.data_handler.mode == "plain":
            text = "plain"
            scalars = "bands"
            vector_name=None
            use_rgba = True

        elif self.data_handler.mode == "parametric":
            text = "parametric"
            scalars = "scalars"
            vector_name=None
            use_rgba = False

        elif self.data_handler.mode == "property_projection":
            
            use_rgba = False

            if self.data_handler.property_name == 'fermi_speed':
                scalars = "Fermi Speed"
                text = "Fermi Speed"
                vector_name=None
            elif self.data_handler.property_name == 'fermi_velocity':
                scalars = "Fermi Velocity Vector_magnitude"
                vector_name = "Fermi Velocity Vector"
                text = "Fermi Speed"
            elif self.data_handler.property_name == 'harmonic_effective_mass':
                scalars = "Harmonic Effective Mass"
                text = "Harmonic Effective Mass"
                vector_name=None
            else:
                print("Please select a property")
        elif self.data_handler.mode == 'spin_texture':
            text = "Spin Texture"
            use_rgba = False
            scalars = "spin_magnitude"
            vector_name = 'spin'

        self.data_handler.text=text
        self.data_handler.scalars_name=scalars
        self.data_handler.vector_name=vector_name
        self.data_handler.use_rgba=use_rgba

    def _normalize_data(self,surface,scalars_name):
        x=surface[scalars_name]
        vmin=self.plotting_options['surface_clim']['value'][0]
        vmax=self.plotting_options['surface_clim']['value'][1]
        x_max=x.max()
        x_min=x.min()
        x_norm =  x_min + ((x - vmin) * (vmax - x_min)) / (x_max - x_min)
        return x_norm

    def _custom_isoslider_callback(self, value):
            res = int(value)
            closest_idx = find_nearest(self.energy_values, res)
            surface=self.e_surfaces[closest_idx]
            if self.plotting_options['surface_color']['value']:
                self.plotter.add_mesh(surface,
                                    name='iso_surface',
                                    color=self.plotting_options['surface_color']['value'],
                                    opacity=self.plotting_options['surface_opacity']['value'],)
            else:
                if self.plotting_options['surface_clim']['value']:
                    self._normalize_data(surface,scalars_name=self.data_handler.scalars_name)
                self.plotter.add_mesh(surface,
                                    name='iso_surface',
                                    scalars=self.data_handler.scalars_name,
                                    cmap=self.plotting_options['surface_cmap']['value'],
                                    clim=self.plotting_options['surface_clim']['value'],
                                    show_scalar_bar=False,
                                    opacity=self.plotting_options['surface_opacity']['value'],
                                    rgba=self.data_handler.use_rgba)

            if self.data_handler.mode != "plain":
                self.add_scalar_bar(name=self.data_handler.scalars_name)
            
            if self.data_handler.scalars_name=="spin_magnitude" or self.data_handler.scalars_name=="Fermi Velocity Vector_magnitude":
                arrows = surface.glyph(orient=self.data_handler.vector_name,
                                            scale=self.plotting_options['texture_scale']['value'] ,
                                            factor=self.plotting_options['texture_size']['value'])
                
                if self.plotting_options['texture_color']['value'] is None:
                    self.plotter.add_mesh(arrows,
                                          name='iso_texture',
                                          scalars=self.data_handler.scalars_name, 
                                            cmap=self.plotting_options['texture_cmap']['value'], 
                                            show_scalar_bar=False,
                                            opacity=self.plotting_options['texture_opacity']['value'])
                else:
                    self.plotter.add_mesh(arrows,
                                          name='iso_texture',
                                          scalars=self.data_handler.scalars_name, 
                                        color=self.plotting_options['texture_color']['value'],
                                        show_scalar_bar=False,
                                        opacity=self.plotting_options['texture_opacity']['value'])
                    

            return None

    def _add_custom_mesh_slice(self,mesh,
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

        line_width=self.plotting_options['cross_section_slice_linewidth']['value']
        #################################################################
        # Following code handles the plane clipping
        #################################################################

        name = kwargs.get('name', mesh.memory_address)


        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.plotter.add_mesh(mesh.outline(), name=name+"outline", 
                              opacity=0.0, 
                              line_width=line_width,
                              show_scalar_bar=False, 
                              rgba=self.data_handler.use_rgba)

        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

    
        self.plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pv.wrap(alg.GetOutput())
        self.plotter.plane_sliced_meshes.append(plane_sliced_mesh)
        

        def callback_plane(normal, origin):
            # create the plane for clipping
            
            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            # plotter.add_mesh(plane_sliced_mesh, name=name+"outline", opacity=0.0, line_width=line_width,show_scalar_bar=False, rgba=options_dict['use_rgba'])
            if self.data_handler.vector_name:
                arrows = plane_sliced_mesh.glyph(orient=self.data_handler.vector_name, scale=False, factor=0.1)
                
                if arrow_color is not None:
                    self.plotter.add_mesh(arrows, color=arrow_color, show_scalar_bar=False,name='arrows')
                else:
                    self.plotter.add_mesh(arrows, cmap=cmap, show_scalar_bar=False,name='arrows')


        self.plotter.add_plane_widget(callback=callback_plane, bounds=mesh.bounds,
                              factor=1.25, normal='x',
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=origin,
                              normal_rotation=normal_rotation)
        

        actor = self.plotter.add_mesh(plane_sliced_mesh,show_scalar_bar=False, 
                                      line_width=line_width,
                                      rgba=self.data_handler.use_rgba, 
                                      **kwargs)
        self.plotter.plane_widgets[0].SetNormal(normal)

        # Call the callback to update scene
        plane_origin = self.plotter.plane_widgets[0].GetOrigin()
        plane_normal = self.plotter.plane_widgets[0].GetNormal()
        callback_plane(normal=plane_normal, origin=plane_origin)
        return actor

    def _add_custom_box_slice_widget(self,
                                    mesh,
                                    show_cross_section_area:bool=False,
                                    line_width:float=5.0,
                                    normal=(1,0,0), 
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
                                    
                                    # box widget options
                                    invert=False,
                                    rotation_enabled=True,
                                    box_widget_color='black',
                                    box_outline_translation=True,
                                    merge_points=True,
                                    crinkle=False,
                                    interaction_event='end',
                                    **kwargs):
        
        #################################################################
        # Following code handles the box clipping
        #################################################################
        
        line_width=self.plotting_options['cross_section_slice_linewidth']['value']

        mesh = pv.PolyData(mesh)

        mesh, algo = algorithm_to_mesh_handler(
            add_ids_algorithm(mesh, point_ids=False, cell_ids=True)
        )
        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))

        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        self.plotter.add_mesh(mesh.outline(), name=f"{name}-outline", opacity=0.0)

        port = 1 if invert else 0

        clipper = vtk.vtkBoxClipDataSet()
        if not merge_points:
            # vtkBoxClipDataSet uses vtkMergePoints by default
            clipper.SetLocator(vtk.vtkNonMergingPointLocator())
        set_algorithm_input(clipper, algo)
        clipper.GenerateClippedOutputOn()

        if crinkle:
            crinkler = crinkle_algorithm(clipper.GetOutputPort(port), algo)
            box_clipped_mesh = _get_output(crinkler)
        else:
            box_clipped_mesh = _get_output(clipper, oport=port)

        self.plotter.box_clipped_meshes.append(box_clipped_mesh)

        def callback_box(planes):
            bounds = []
            for i in range(planes.GetNumberOfPlanes()):
                plane = planes.GetPlane(i)
                bounds.append(plane.GetNormal())
                bounds.append(plane.GetOrigin())

            clipper.SetBoxClip(*bounds)
            clipper.Update()
            if crinkle:
                clipped = pv.wrap(crinkler.GetOutputDataObject(0))
            else:
                clipped = _get_output(clipper, oport=port)
            box_clipped_mesh.shallow_copy(clipped)


            # Update plane widget after updating box widget
            plane_origin = self.plotter.plane_widgets[0].GetOrigin()
            plane_normal = self.plotter.plane_widgets[0].GetNormal()
            callback_plane(normal=plane_normal, origin=plane_origin)


        #################################################################
        # Following code handles the plane clipping
        #################################################################

        clipped_box_mesh =  self.plotter.box_clipped_meshes[0]
        name = kwargs.get('name', clipped_box_mesh.memory_address)

        # print(name)
        # if kwargs.get('scalars', mesh.active_scalars_name) != 'spin':
            
        rng = clipped_box_mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        clipped_box_mesh.set_active_scalars(kwargs.get('scalars', clipped_box_mesh.active_scalars_name))

        self.plotter.add_mesh(clipped_box_mesh.outline(), 
                              name=name+"outline", 
                              opacity=0.0, 
                              line_width=0.0,
                              show_scalar_bar=False, 
                              rgba=self.data_handler.use_rgba)

        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(clipped_box_mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

    
        self.plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pv.wrap(alg.GetOutput())
        self.plotter.plane_sliced_meshes.append(plane_sliced_mesh)
        
        if show_cross_section_area:
            user_slice = self.plotter.plane_sliced_meshes[0]
            surface = user_slice.delaunay_2d()
            self.plotter.add_text(f"Cross sectional area : {surface.area:.4f}"+" Ang^-2", color = 'black')

  
        def callback_plane(normal, origin):
            # create the plane for clipping
            
            plane = generate_plane(normal, origin)
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            # plotter.add_mesh(plane_sliced_mesh, name=name+"outline", opacity=0.0, line_width=line_width,show_scalar_bar=False, rgba=options_dict['use_rgba'])
            if self.data_handler.vector_name:
                arrows = plane_sliced_mesh.glyph(orient=self.data_handler.vector_name, scale=False, factor=0.1)
                
                if arrow_color is not None:
                    self.plotter.add_mesh(arrows, color=arrow_color, show_scalar_bar=False,name='arrows')
                else:
                    self.plotter.add_mesh(arrows, cmap=cmap, show_scalar_bar=False,name='arrows')

            
            if show_cross_section_area:
                user_slice = self.plotter.plane_sliced_meshes[0]
                surface = user_slice.delaunay_2d()
                text = f"Cross sectional area : {surface.area:.4f}"+" Ang^-2"

                self.plotter.textActor.SetText(2, text)


        self.plotter.add_plane_widget(callback=callback_plane, bounds=mesh.bounds,
                              factor=1.25, normal=normal,
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=origin,
                              normal_rotation=normal_rotation)
        

        self.plotter.add_box_widget(
            callback=callback_box,
            bounds=mesh.bounds,
            factor=1.25,
            rotation_enabled=rotation_enabled,
            use_planes=True,
            color=box_widget_color,
            outline_translation=box_outline_translation,
            interaction_event=interaction_event,
        )

       
        actor = self.plotter.add_mesh(plane_sliced_mesh,
                                      show_scalar_bar=False, 
                                      line_width=line_width,
                                      rgba=self.data_handler.use_rgba, 
                                      **kwargs)
        self.plotter.plane_widgets[0].SetNormal(normal)
    
        # # Call the callback to update scene
        plane_origin = self.plotter.plane_widgets[0].GetOrigin()
        plane_normal = self.plotter.plane_widgets[0].GetNormal()
        callback_plane(normal=plane_normal, origin=plane_origin)
        return actor


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx