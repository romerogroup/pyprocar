# -*- coding: utf-8 -*-

import numpy as np
import pyvista
from matplotlib import colors as mpcolors
from matplotlib import cm
from .fermisurface3d import boolean_add
from .fermisurface3d import FermiSurface3D
from .splash import welcome
from .fermisurface3d import FermiSurface3D
from .utilsprocar import UtilsProcar
from .procarparser import ProcarParser
from .procarselect import ProcarSelect

def fermi3D(procar='PROCAR',
            outcar='OUTCAR',
            fermi=None,
            bands=None, 
            interpolation_factor=1,
            mode="plain",
            colors=None,
            background_color='white',
            save_colors=None,
            cmap='viridis',
            atoms=None,
            orbitals=None,
            spin=None,
            projection_accuracy='normal',
            spin_texture=False,
            fermi_shift=0,
            arrow_color=None,
            code='vasp',
            vmin=0,
            vmax=1,
            save3d=None,
            perspective=True,
            save2d=False,
            camera_pos=[1,1,1],
            widget=None,
            show=True,
            ):
    
    """
    
    """
    
    
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

    if mode == 'parametric':
        if orbitals is None:
            orbitals = [-1]
        if atoms is None:
            atoms = [-1]
        if spin is None:
            spin = [0]
        data.selectIspin(spin)
        data.selectAtoms(atoms, fortran=False)
        data.selectOrbital(orbitals)
        spd = data.spd
        for iband in bands:
           surface = FermiSurface3D(kpoints=data.kpoints,
                                    band=data.bands[:,iband],
                                    spd=spd[:,iband],
                                    fermi=e_fermi+fermi_shift,
                                    reciprocal_lattice=reciprocal_lattice,
                                    interpolation_factor=interpolation_factor,
                                    projection_accuracy=projection_accuracy)
           if surface.verts is not None:
               surfaces.append(surface)       
    elif mode == 'plain':
        for iband in bands:
            surface = FermiSurface3D(kpoints=data.kpoints,
                                        band=data.bands[:,iband],
                                        fermi=e_fermi+fermi_shift,
                                        reciprocal_lattice=reciprocal_lattice,
                                        interpolation_factor=interpolation_factor,
                                        projection_accuracy=projection_accuracy)
            if surface.verts is not None:
                surfaces.append(surface)

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
            if mode == 'plain':
                p.add_mesh(surfaces[isurface].pyvista_obj,
                           color=colors[isurface])
                text='Colors'
            elif mode =='parametric':
                p.add_mesh(surfaces[isurface].pyvista_obj,cmap=cmap)
                p.remove_scalar_bar()
                text='Projection'

                
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
        # p.set_position(camera_pos)
        p.show(cpos=camera_pos,screenshot=save2d)
        # p.screenshot('1.png')
        # p.save_graphic('1.pdf')
    
    if save_colors is None:
        for i in range(nsurface):
            if surfaces[i].scalars is None:
                surfaces[i].set_scalars([scalars[i]]*surfaces[i].nfaces)
    s = boolean_add(surfaces)
    s.set_color_with_cmap(cmap=cmap,vmin=vmin,vmax=vmax)
    
    # s.pyvista_obj.plot()
    if save3d is not None:
        extention=save3d.split('.')[-1]
        s.export(save3d,extention)
    return s,surfaces