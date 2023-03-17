#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:14:17 2020

@author: petavazohi
"""

import numpy as np
import pyvista
from matplotlib import colors as mpcolors
from matplotlib import cm
from ..core import boolean_add
from ..core import FermiSurface3D
from ..utils import welcome
from ..utils import UtilsProcar
from ..io import ProcarParser
from ..core import ProcarSelect

def spin_asymmetry(procar='PROCAR',
                   outcar='OUTCAR',
                   fermi=None,
                   bands=None, 
                   interpolation_factor=1,
                   mode="plain",
                   supercell=[1,1,1],
                   colors=None,
                   background_color='white',
                   save_colors=None,
                   cmap='viridis',
                   atoms=None,
                   orbitals=None,
                   spin=None,
                   spin_texture=False,
                   arrow_color=None,
                   arrow_size=0.015,
                   only_spin=False,
                   fermi_shift=0,
                   projection_accuracy='normal',
                   code='vasp',
                   vmin=0, 
                   vmax=1,
                   savegif=None,
                   savemp4=None,
                   save3d=None,
                   perspective=True,
                   save2d=False,
                   camera_pos=[1,1,1],
                   widget=None,
                   show=True,
            ):
    
    
    
    
    welcome()
    
    
    if show:
        p = pyvista.Plotter()
    
    if code=='vasp':
        outcarparser = UtilsProcar()
        if fermi is None:
            e_fermi = outcarparser.FermiOutcar(outcar)
        else:
            e_fermi = fermi
        reciprocal_lattice = outcarparser.RecLatOutcar(outcar)
        procarFile = ProcarParser()
        procarFile.readFile(procar, False)
        data = ProcarSelect(procarFile, deepCopy=True)
    if bands is None:
        bands = np.arange(len(data.bands[0,:]))
    surfaces = []
    
        
        
    spd = []        
    if mode == 'parametric':
        if orbitals is None:
            orbitals = [-1]
        if atoms is None:
            atoms = [-1]
        
        spin = [0,1]
        data.selectIspin(spin)
        data.selectAtoms(atoms, fortran=False)
        data.selectOrbital(orbitals)

        for iband in bands:
            spd.append(data.spd[:,iband])
    else :
        for iband in bands:
            spd.append(None)

    spd_spin=[]
    
    if spin_texture:
        dataX = ProcarSelect(procarFile, deepCopy=True)
        dataY = ProcarSelect(procarFile, deepCopy=True)
        dataZ = ProcarSelect(procarFile, deepCopy=True)

        dataX.kpoints = data.kpoints
        dataY.kpoints = data.kpoints
        dataZ.kpoints = data.kpoints

        dataX.spd = data.spd
        dataY.spd = data.spd
        dataZ.spd = data.spd

        dataX.bands = data.bands
        dataY.bands = data.bands
        dataZ.bands = data.bands

        dataX.selectIspin([1])
        dataY.selectIspin([2])
        dataZ.selectIspin([3])

        dataX.selectAtoms(atoms, fortran=False)
        dataY.selectAtoms(atoms, fortran=False)
        dataZ.selectAtoms(atoms, fortran=False)

        dataX.selectOrbital(orbitals)
        dataY.selectOrbital(orbitals)
        dataZ.selectOrbital(orbitals)
        for iband in bands:
            spd_spin.append([dataX.spd[:,iband],dataY.spd[:,iband],dataZ.spd[:,iband]])
    else:
        for iband in bands:
            spd_spin.append(None)
    counter = 0
    for iband in bands:
        print("Trying to extract isosurface for band %d"%iband)
        surface = FermiSurface3D(kpoints=data.kpoints,
                                 band=data.bands[:,iband],
                                 spd=spd[counter],
                                 spd_spin=spd_spin[counter],
                                 fermi=e_fermi+fermi_shift,
                                 reciprocal_lattice=reciprocal_lattice,
                                 interpolation_factor=interpolation_factor,
                                 projection_accuracy=projection_accuracy,
                                 supercell=supercell)
        if surface.verts is not None:
            surfaces.append(surface)       
        counter+=1

        
        
    nsurface = len(surfaces)
    norm = mpcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap)
    scalars = np.arange(nsurface+1)/nsurface
    if colors is None:
        colors = np.array([cmap(norm(x)) for x in (scalars)]).reshape(-1,4)
    
    
    if show or save2d:
        # sargs = dict(interactive=True) 
        p.add_mesh(surfaces[0].brillouin_zone.pyvista_obj,
                    style='wireframe',line_width=3.5,color='black')
        for isurface in range(nsurface):
            if not only_spin:
                if mode == 'plain':
                    p.add_mesh(surfaces[isurface].pyvista_obj,
                               color=colors[isurface])
                    text='Colors'
                elif mode =='parametric':
                    p.add_mesh(surfaces[isurface].pyvista_obj,
                               cmap=cmap,
                               clim=[vmin,vmax])
                    p.remove_scalar_bar()
                    text='Projection'
            else:
                text='Spin Texture'
            if spin_texture:
                # Example dataset with normals
                # create a subset of arrows using the glyph filter
                arrows = surfaces[isurface].pyvista_obj.glyph(orient="vectors",factor=arrow_size)

                if arrow_color is None:
                    p.add_mesh(arrows,
                               cmap=cmap,
                               clim=[vmin,vmax])
                    p.remove_scalar_bar()
                else:
                    p.add_mesh(arrows,
                               color=arrow_color)

        p.add_scalar_bar(title=text,
                         n_labels=6,
                         italic=False,
                         bold=False,
                         title_font_size=None,
                         label_font_size=None,
                         position_x=0.9,
                         position_y=0.01,
                         color='black')
                
        p.add_axes(xlabel='Kx',
                   ylabel='Ky',
                   zlabel='Kz',
                   line_width=6,
                   labels_off=False)

        if not perspective :
            p.enable_parallel_projection()
        p.set_background(background_color)
        p.set_position(camera_pos)
        p.show(cpos=camera_pos,screenshot=save2d)
        # p.screenshot('1.png')
        # p.save_graphic('1.pdf')
        if savegif is not None:
            path = p.generate_orbital_path(n_points=36)
            p.open_gif(savegif)
            p.orbit_on_path(path)#,viewup=camera_pos)
        if savemp4:
            path = p.generate_orbital_path(n_points=36)
            p.open_movie(savemp4)
            p.orbit_on_path(path)#,viewup=camera_pos)
            # p.close()        
    if save_colors is None:
        for i in range(nsurface):
            if surfaces[i].scalars is None:
                surfaces[i].set_scalars([scalars[i]]*surfaces[i].nfaces)
    s = boolean_add(surfaces)
    s.set_color_with_cmap(cmap=cmap,vmin=vmin,vmax=vmax)
    # s.pyvista_obj.plot()
    # s.trimesh_obj.show()
    
    if save3d is not None:
        extention=save3d.split('.')[-1]
        s.export(save3d,extention)
    return s,surfaces
