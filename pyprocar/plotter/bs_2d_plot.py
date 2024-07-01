import os
import copy
import yaml
from typing import List, Tuple

import numpy as np
import pyvista as pv
from matplotlib import colors as mpcolors
from matplotlib import cm
from PIL import Image
from pyvista.core.filters import _get_output  # avoids circular import

from pyprocar import io
from pyprocar.core import BandStructure2D
from pyprocar.utils import ROOT, ConfigManager



class BandStructure2DataHandler:

    def __init__(self, ebs, fermi_tolerance=0.1, **kwargs):
        self.initial_ebs=copy.copy(ebs)
        self.ebs = copy.copy(ebs)
        self.fermi_tolerance = fermi_tolerance
        self.mode=None
        # Other instance variables...

        config_manager=ConfigManager(os.path.join(ROOT,'pyprocar','cfg','band_structure_2d.yml'))
        config_manager.update_config(kwargs)  
        self.config=config_manager.get_config()

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
            if orbitals is None and self.ebs.projected is not None:
                orbitals = np.arange(self.ebs.norbitals, dtype=int)
            if atoms is None and self.ebs.projected is not None:
                atoms = np.arange(self.ebs.natoms, dtype=int)

            if self.ebs.is_non_collinear:
                projected = self.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=True)
            else:
                projected = self.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

            for ispin in self.spin_pol:
                spin_bands_projections = []
                for iband in bands_to_keep:
                    spin_bands_projections.append(projected[:,iband,ispin])
                spd.append( spin_bands_projections)
            spd = np.array(spd).T
            spins = np.arange(spd.shape[2])
        else:
            spd = np.zeros(shape = (self.ebs.nkpoints,len(bands_to_keep),len(spins)))
        spd_spin = []
        if spin_texture:
            ebs = copy.deepcopy(self.ebs)
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

    def get_surface_data(self,property_name=None):
        
        if self.mode is None:
            raise "You must call process data function before get_surface"
        if property_name:
            current_emplemented_properties = ['band_velocity', 'band_speed' ,'harmonic_effective_mass']
            if property_name not in current_emplemented_properties:
                tmp=f"You must choose one of the following properies : {current_emplemented_properties}"
                raise ValueError(tmp)
                
        band_surfaces = []
        spins_band_index=[]
        spins_index=[]
        for ispin, spin in enumerate(self.spin_pol):
            ebs=copy.copy(self.ebs)
            
            band_structure_2D = BandStructure2D(
                                            ebs,
                                            spin,
                                            interpolation_factor=self.config['interpolation_factor']['value'],
                                            projection_accuracy=self.config['projection_accuracy']['value'],
                                            supercell=self.config['supercell']['value'],
                                            zlim=self.config['energy_lim']['value']
                                        )
            self.property_name=property_name
            if self.property_name=='band_speed':
                band_structure_2D.project_band_speed(band_speed=ebs.fermi_speed[...,ispin])
            elif self.property_name=='band_velocity':
                band_structure_2D.project_band_velocity(band_velocity=ebs.fermi_velocity[...,ispin,:])
            elif self.property_name=='harmonic_effective_mass':
                band_structure_2D.project_harmonic_effective_mass(harmonic_effective_mass=ebs.harmonic_average_effective_mass[...,ispin])
            if self.mode =='parametric':
                band_structure_2D.project_atomic_projections(self.spd[...,ispin])

            if self.mode =='spin_texture':
                band_structure_2D.project_spin_texture_atomic_projections(self.spd_spin)

            if self.config['extended_zone_directions']['value']:
                band_structure_2D.extend_surface(extended_zone_directions=self.config['extended_zone_directions']['value'])
            band_surfaces.append(band_structure_2D)

                
            if ispin==1:
                spin_band_index = list(band_structure_2D.point_data['band_index'] + len(np.unique(spin_band_index)) )
                spin_band_index.reverse()
            else:
                spin_band_index = list(band_structure_2D.point_data['band_index'])
                spin_band_index.reverse()
            spins_band_index.extend(spin_band_index)

            spin_index=[ispin]*len(band_structure_2D.points)
            spin_index.reverse()
            spins_index.extend(spin_index)


        # These reverese are used 
        # because when you combine two meshes the point_data 
        # and points are prepended to each other, so extend would be backwards
        spins_band_index.reverse()
        spins_index.reverse()

        band_surface=None
        for i,surface in enumerate(band_surfaces):
            if i == 0:
                band_surface=surface
            else:
                band_surface+=surface

        band_surface.point_data['band_index']= np.array(spins_band_index)
        band_surface.point_data['spin_index']= np.array(spins_index)
        
        self.band_surface=band_surface
        return self.band_surface
    
class BandStructure2DVisualizer:

    def __init__(self, data_handler=None,**kwargs):

        self.data_handler = data_handler
        self.plotter=pv.Plotter()

        config_manager=ConfigManager(os.path.join(ROOT,'pyprocar','cfg','band_structure_2d.yml'))
        config_manager.update_config(kwargs)  
        self.config=config_manager.get_config()

        self._setup_plotter()

    def add_scalar_bar(self,name):
        if self.config['scalar_bar_title']['value']:
            name=self.config['scalar_bar_title']['value']
        else:
            name=name
            
        if self.config['add_scalar_bar']['value']:
            self.plotter.add_scalar_bar(
                    title=name,
                    n_labels=self.config['scalar_bar_labels']['value'],
                    italic=self.config['scalar_bar_italic']['value'],
                    bold=self.config['scalar_bar_bold']['value'],
                    title_font_size=self.config['scalar_bar_title_font_size']['value'],
                    label_font_size=self.config['scalar_bar_label_font_size']['value'],
                    position_x=self.config['scalar_bar_position_x']['value'],
                    position_y=self.config['scalar_bar_position_y']['value'],
                    color=self.config['scalar_bar_color']['value'])
        
    def add_axes(self):
        if self.config['add_axes']['value']:
            self.plotter.add_axes(
                    xlabel=self.config['x_axes_label']['value'], 
                    ylabel=self.config['y_axes_label']['value'], 
                    zlabel=self.config['z_axes_label']['value'],
                    color=self.config['axes_label_color']['value'],
                    line_width=self.config['axes_line_width']['value'],
                labels_off=False)

    def add_brillouin_zone(self,surface):
       self.plotter.add_mesh(
                surface.brillouin_zone,
                style=self.config['brillouin_zone_style']['value'],
                line_width=self.config['brillouin_zone_line_width']['value'],
                color=self.config['brillouin_zone_color']['value'],
                opacity=self.config['brillouin_zone_opacity']['value']
            )
    
    def add_surface(self,surface):

        surface=self._setup_band_colors(surface)
        if self.config['surface_spinpol_colors']['value']:
            spin_colors=[]
            for spin_index in surface.point_data['spin_index']:
                if spin_index == 0:
                    spin_colors.append(self.config['surface_spinpol_colors']['value'][0])
                else:
                    spin_colors.append(self.config['surface_spinpol_colors']['value'][1])
            surface.point_data['spin_colors']=spin_colors
            self.plotter.add_mesh(surface,
                                scalars='spin_colors',
                                cmap=self.config['surface_cmap']['value'],
                                clim=self.config['surface_clim']['value'],
                                show_scalar_bar=False,
                                opacity=self.config['surface_opacity']['value'],)
                                # rgba=self.data_handler.use_rgba)
            

        elif self.config['surface_color']['value']:
            self.plotter.add_mesh(surface,
                                color=self.config['surface_color']['value'],
                                opacity=self.config['surface_opacity']['value'],)
        else:
            if self.config['surface_clim']['value']:
                self._normalize_data(surface,scalars_name=self.data_handler.scalars_name)

            self.plotter.add_mesh(surface,
                                scalars=self.data_handler.scalars_name,
                                cmap=self.config['surface_cmap']['value'],
                                clim=self.config['surface_clim']['value'],
                                show_scalar_bar=False,
                                opacity=self.config['surface_opacity']['value'],
                                rgba=self.data_handler.use_rgba)
            
    def add_texture(self,
                    surface,
                    scalars_name, 
                    vector_name):
        
        arrows=None
        if scalars_name=="spin_magnitude" or scalars_name=="Band Velocity Vector_magnitude":
            arrows = surface.glyph(orient=vector_name,
                                         scale=self.config['texture_scale']['value'] ,
                                         factor=self.config['texture_size']['value'])
            if self.config['texture_color']['value'] is None:
                self.plotter.add_mesh(arrows,scalars=scalars_name, 
                                      cmap=self.config['texture_cmap']['value'], 
                                      show_scalar_bar=False,
                                      opacity=self.config['texture_opacity']['value'])
            else:
                self.plotter.add_mesh(arrows,scalars=scalars_name, 
                                      color=self.config['texture_color']['value'],
                                      show_scalar_bar=False,
                                      opacity=self.config['texture_opacity']['value'])
                
        return arrows
    
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
            self.plotter.show(cpos=self.config['plotter_camera_pos']['value'], 
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
            if self.config['texture_color']['value'] is not None:
                p.add_mesh(arrows, color=self.config['texture_color']['value'], show_scalar_bar=False,name='arrows')
            else:
                p.add_mesh(arrows, 
                           cmap=self.config['texture_cmap']['value'], 
                           show_scalar_bar=False,name='arrows')
            p.add_mesh(slice_2d,
                       line_width=self.config['cross_section_slice_line_width']['value'])
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
                    show_cross_section_area=self.config['cross_section_slice_show_area'],
                    normal=slice_normal,
                    origin=slice_origin,
                    )

        if show:
            self.plotter.show(cpos=self.config['plotter_camera_pos']['value'], 
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
            if self.config['texture_color']['value'] is not None:
                p.add_mesh(arrows, color=self.config['texture_color']['value'], show_scalar_bar=False,name='arrows')
            else:
                p.add_mesh(arrows, 
                           cmap=self.config['texture_cmap']['value'], 
                           show_scalar_bar=False,name='arrows')
            p.add_mesh(slice_2d,
                       line_width=self.config['cross_section_slice_line_width']['value'])
            p.remove_scalar_bar()
            # p.set_background(background_color)
            p.view_vector(normal_vec)
            p.show(screenshot=save_2d_slice,interactive=False)

    def add_grid(self,z_label=None):
        if self.config['grid']['value']:
            
            if z_label is None:
                z_label=self.config['grid_ztitle']['value']

            self.plotter.show_grid(
                                ytitle=self.config['grid_ytitle']['value'],
                                xtitle=self.config['grid_xtitle']['value'],
                                ztitle=z_label,
                                font_size=self.config['grid_font_size']['value'])

    def add_fermi_plane(self, value:float):
        if self.config['add_fermi_plane']['value']:
            plane = pv.Plane(center=(0,0,value),
                            i_size=self.config['fermi_plane_size']['value'],
                            j_size=self.config['fermi_plane_size']['value'])

            self.plotter.add_mesh(plane,#.extract_feature_edges(),
                                  color=self.config['fermi_plane_color']['value'],
                                  render_lines_as_tubes=True,
                                  opacity=self.config['fermi_plane_opacity']['value'],
                                  line_width=5)
            if self.config['fermi_plane_opacity']['value']:
                poly = pv.PolyData(np.array(self.config['fermi_text_position']['value']))
                poly['labels']=['E$_{F}$' for i in range(poly.n_points)]
                self.plotter.add_point_labels(poly, "labels",font_size=36,show_points=False)
            
    def set_background_color(self):
        self.plotter.set_background(self.config['background_color']['value'])

    def close(self):
        self.plotter.close()

    def show(self,filename=None):
        if filename:
            file_extentions = filename.split()
            if self.config['plotter_offscreen']:
                self.plotter.off_screen = True
                self.plotter.show( cpos=self.config['plotter_camera_pos']['value'],auto_close=False)  
                self.plotter.show(screenshot=filename)
            else:
                image=self.plotter.show( cpos=self.config['plotter_camera_pos']['value'],screenshot=True)  
                im = Image.fromarray(image)
                im.save(filename)
        else:
            self.plotter.show(cpos=self.config['plotter_camera_pos']['value'])
        
    def save_gif(self,filename):
        path = self.plotter.generate_orbital_path(n_points=self.config['orbit_gif_n_points']['value'])
        self.plotter.open_gif(filename)
        self.plotter.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=self.config['orbit_gif_step']['value'])
    
    def save_mp4(self,filename):
        path = self.plotter.generate_orbital_path(n_points=self.config['orbit_mp4_n_points']['value'])
        self.plotter.open_movie(filename)
        self.plotter.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=self.config['orbit_mp4_step']['value'])
    
    def save_mesh(self,filename,surface):
        pv.save_meshio(filename, surface)
    
    def _setup_band_colors(self,fermi_surface):
        band_colors = self.config['surface_bands_colors']['value']
        if self.config['surface_bands_colors']['value'] == []:
            band_colors=self._generate_band_colors(fermi_surface)

        fermi_surface = self._apply_fermi_surface_band_colors(fermi_surface,band_colors)
        return fermi_surface

    def _apply_fermi_surface_band_colors(self,fermi_surface,band_colors):
        unique_band_index = np.unique(fermi_surface.point_data['band_index'])
        if len(band_colors) != len(unique_band_index):
            raise "You need to list the number of colors as there are bands that make up the surface"
        
        surface_band_colors=[]
        band_colors_index_map={unique_index:i for i,unique_index in enumerate(unique_band_index)}
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
            band_surface_index=band_colors_index_map[band_surface_index]
            band_color = surface_band_colors[band_surface_index]
            band_colors.append(band_color)
        fermi_surface.point_data['bands']=band_colors

        return fermi_surface
    
    def _generate_band_colors(self,fermi_surface):
        # Generate unique rgba values for the bands
        unique_band_index = np.unique(fermi_surface.point_data['band_index'])
        nsurface = len(unique_band_index)
        norm = mpcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap(self.config['surface_cmap']['value'])
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

            if self.data_handler.property_name == 'band_speed':
                scalars = "Band Speed"
                text = "Band Speed"
                vector_name=None
            elif self.data_handler.property_name == 'band_velocity':
                scalars = "Band Velocity Vector_magnitude"
                vector_name = "Band Velocity Vector"
                text = "Band Speed"
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
        vmin=self.config['surface_clim']['value'][0]
        vmax=self.config['surface_clim']['value'][1]
        x_max=x.max()
        x_min=x.min()
        x_norm =  x_min + ((x - vmin) * (vmax - x_min)) / (x_max - x_min)
        return x_norm

    def clip_broullin_zone(self,surface):
        if self.config['clip_brillouin_zone']['value']:
            
            for normal,center in zip(surface.brillouin_zone.face_normals, surface.brillouin_zone.centers):
                
                center_with_factor = center
                center_with_factor[0] = center_with_factor[0]*self.config['clip_brillouin_zone_factor']['value']
                center_with_factor[1] = center_with_factor[1]*self.config['clip_brillouin_zone_factor']['value']
                surface.clip(origin=center_with_factor, normal=normal, inplace=True)
        return surface
    
    def update_config(self, config_dict):
        for key,value in config_dict.items():
            self.config[key]['value']=value