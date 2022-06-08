__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import sys
from typing import List, Tuple
import copy

import numpy as np
from matplotlib import colors as mpcolors
from matplotlib import cm
import vtk
import pyvista
from pyvista.utilities import NORMALS, generate_plane, get_array, try_callback

# from .core.surface import boolean_add
from ..fermisurface3d import FermiSurface3D
from ..splash import welcome
from ..utilsprocar import UtilsProcar
from ..io.procarparser import ProcarParser
from ..procarselect import ProcarSelect
from ..io.bxsf import BxsfParser
from ..io.frmsf import FrmsfParser
from ..io.qeparser import QEFermiParser
from ..io.lobsterparser import LobsterFermiParser
from ..io.abinitparser import AbinitParser
from .. import io

np.set_printoptions(threshold=sys.maxsize)

# TODO update the parsing section
# TODO separate slicing functionality to new function
# TODO separate iso-slider functionality to new function. isoslider still experimental

def fermi3D(
    procar:str="PROCAR",
    outcar:str="OUTCAR",
    poscar:str="POSCAR",
    dirname:str="",
    infile:str="in.bxsf",
    abinit_output:str=None,
    fermi:float=None,
    bands:List[int]=None,
    interpolation_factor:int=1,
    mode:str="plain",
    supercell:List[int]=[1, 1, 1],
    extended_zone_directions:List[List[int]] = None,
    colors: List[str] or List[Tuple[float,float,float]]=None,
    background_color:str="white",
    save_colors:bool=False,
    cmap:str="jet",
    atoms:List[int]=None,
    orbitals:List[int]=None,
    calculate_fermi_speed: bool=False,
    calculate_fermi_velocity: bool=False,
    calculate_effective_mass: bool=False,
    spins:List[int]=None,
    spin_texture: bool=False,
    arrow_color=None,
    arrow_size: float=0.015,
    only_spin: bool=False,
    fermi_shift:float=0,
    fermi_tolerance:float=0.1,
    projection_accuracy:str="normal",
    code:str="vasp",
    vmin:float=0,
    vmax:float=1,
    savegif:str=None,
    savemp4:str=None,
    save3d:str=None,
    save_meshio:bool=False,
    perspective:bool=True,
    save2d:bool=False,
    show_slice:bool=False,
    slice_normal: Tuple[float,float,float]=(1,0,0),
    slice_origin: Tuple[float,float,float]=(0,0,0),
    show_cross_section_area: bool=False,
    iso_slider: bool=False,
    iso_range: float=2,
    iso_surfaces: int=10,
    camera_pos:List[float]=[1, 1, 1],
    widget:bool=False,
    show:bool=True,
    repair:bool=True,
):
    """
    Parameters
    ----------
    procar : str, optional (default ``'PROCAR'``)
        Path to the PROCAR file of the simulation
        e.g. ``procar='~/MgB2/fermi/PROCAR'``
    outcar : str, optional (default ``'OUTCAR'``)
        Path to the OUTCAR file of the simulation
        e.g. ``outcar='~/MgB2/fermi/OUTCAR'``
    abinit_output : str, optional (default ``None``)
        Path to the Abinit output file
        e.g. ``outcar='~/MgB2/abinit.out'``
    infile : str, optional (default ``infile = in.bxsf'``)
        This is the path in the input bxsf file
        e.g. ``infile = ni_fs.bxsf'``
    fermi : float, optional (default ``None``)
        Fermi energy at which the fermi surface is created. In other
        words fermi is the isovalue at which the fermi surface is
        created. If not defined it is read from the OUTCAR file.
        e.g. ``fermi=-5.49``
    bands : list int, optional
        Which bands are going to be plotted in the fermi surface. The
        numbering is based on vasp outputs. If nothing is selected,
        this function will iterate over all the bands and plots the
        ones that cross fermi.
        e.g. ``bands=[14, 15, 16, 17]``
    interpolation_factor : int, optional
        The kpoints grid will increase by this factor and interpolated
        at the new points using Fourier interpolation.
        e.g. If the kgrid is 5x5x5, ``interpolation_factor=4`` will
        lead to a kgrid of 20x20x20
    mode : str, optional (default ``mode='plain'``)
        Defines If the fermi surface will have any projection using
        colormaps or is a plotted with a uniform plain color.
        e.g. ``mode='plain'``, ``mode='parametric'``
    supercell : list int, optional (default ``[1, 1, 1]``)
        If one wants plot more than the 1st brillouin zone, this
        parameter can be used.
        e.g. ``supercell=[2, 2, 2]``
    extended_zone_directions : list of list of size 3, optional (default ``None``)
        If one wants plot more than  brillouin zones in a particular direection, this
        parameter can be used.
        e.g. ``extended_zone_directions=[[1,0,0],[0,1,0],[0,0,1]]``
    colors : list str or list tuples of size 4, optional
        List of colors for each band. If you use tuple, it represents rgba values
        This argument does not work whena 3d file is saved. 
        The colors for when ``save3d`` is used, we
        recomend using qualitative colormaps, as this function will
        automatically choose colors from the colormaps.
        e.g. ``colors=['red', 'blue', 'green']``
            ``colors=[(1,0,0,1), (0,1,0,1), (0,0,1,1)]``
    background_color : str, optional (default ``white``)
        Defines the background color.
        e.g. ``background_color='gray'``
    save_colors : bool, optional (default ``False``)
        In case the plot is saved in 3D and some property of the
        material is projected on the fermi surface, this argument
        allows the projection to be stored in the 3D file.
        e.g. ``save_colors=True``
    cmap : str, optional (default ``jet``)
        The color map used for color coding the projections. ``cmap``
        is only relevant in ``mode='parametric'``. A full list of
        color maps in matplotlib are provided in this web
        page. `https://matplotlib.org/2.0.1/users/colormaps.html
        <https://matplotlib.org/2.0.1/users/colormaps.html>`_
    atoms : list int, optional
        ``atoms`` define the projection of the atoms on the fermi
        surfcae . In other words it selects only the contribution of the
        atoms provided. Atoms has to be a python list(or numpy array)
        containing the atom indices. Atom indices has to be order of
        the input files of DFT package. ``atoms`` is only relevant in
        ``mode='parametric'``. keep in mind that python counting
        starts from zero.
        e.g. for SrVO\ :sub:`3`\  we are choosing only the oxygen
        atoms. ``atoms=[2, 3, 4]``, keep in mind that python counting
        starts from zero, for a **POSCAR** similar to following::
            Sr1 V1 O3
            1.0
            3.900891 0.000000 0.000000
            0.000000 3.900891 0.000000
            0.000000 0.000000 3.900891
            Sr V O
            1 1 3
            direct
            0.500000 0.500000 0.500000 Sr atom 0
            0.000000 0.000000 0.000000 V  atom 1
            0.000000 0.500000 0.000000 O  atom 2
            0.000000 0.000000 0.500000 O  atom 3
            0.500000 0.000000 0.000000 O  atom 4
        if nothing is specified this parameter will consider all the
        atoms present.
    orbitals : list int, optional
        ``orbitals`` define the projection of orbitals on the fermi
        surface. In other words it selects only the contribution of
        the orbitals provided. Orbitals has to be a python list(or
        numpy array) containing the Orbital indices. Orbitals indices
        has to be order of the input files of DFT package. The
        following table represents the indecies for different orbitals
        in **VASP**.
            +-----+-----+----+----+-----+-----+-----+-----+-------+
            |  s  | py  | pz | px | dxy | dyz | dz2 | dxz | x2-y2 |
            +-----+-----+----+----+-----+-----+-----+-----+-------+
            |  0  |  1  |  2 |  3 |  4  |  5  |  6  |  7  |   8   |
            +-----+-----+----+----+-----+-----+-----+-----+-------+
        ``orbitals`` is only relavent in ``mode='parametric'``
        e.g. ``orbitals=[1,2,3]`` will only select the p orbitals
        while ``orbitals=[4,5,6,7,8]`` will select the d orbitals.
        If nothing is specified pyprocar will select all the present
        orbitals.
    calculate_fermi_velocity : bool, optional (default False)
            Boolean value to calculate fermi velocity vectors on the fermi surface. 
            Must be used with mode= "property_projection".
            e.g. ``fermi_velocity_vector=True``
    calculate_fermi_speed : bool, optional (default False)
        Boolean value to calculate magnitude of the fermi velocity on the fermi surface.
        Must be used with mode= "property_projection".
        e.g. ``fermi_velocity=True``
    calculate_effective_mass : bool, optional (default False)
        Boolean value to calculate the harmonic mean of the effective mass on the fermi surface.
        Must be used with mode= "property_projection".
        e.g. ``effective_mass=True``
    spins : list int, optional
        e.g. ``spins=[0]``
    spin_texture : bool, optional (default False)
        In non collinear calculation one can choose to plot the spin
        texture on the fermi surface.
        e.g. ``spin_texture=True``
    arrow_color : str, optional
        Defines the color of the arrows when
        ``spin_texture=True``. The default will select the colors
        based on the color map specified. If arrow_color is selected,
        all arrows will have the same color.
        e.g. ``arrow_color='red'``
    arrow_size : int, optional
        As the name suggests defines the size of the arrows, when spin
        texture is selected.
        e.g. ``arrow_size=3``
    only_spin : bool, optional
        If ``only_spin=True`` is selected, the fermi surface is not
        plotted and only the spins in the spin texture is plotted.
    fermi_shift : float, optional
        This parameter is useful when one wants to plot the iso-surface
        above or belove the fermi level.
        e.g. ``fermi_shift=0.6``
    fermi_tolerance : float = 0.1
        This is used to improve search effiency by doing a prior search selecting band within a tolerance of the fermi energy
    projection_accuracy : str, optional (default ``'normal'``)
        Selected the accuracy of projected properties. ``'normal'`` and
        ``'high'`` are the only two options. ``'normal'`` uses the fast
        but rather inaccurate nearest neighbor interpolation, while
        ``'high'`` uses the more accurate linear interpolation for the
        projection of the properties.
        e.g. ``projection_accuracy='high'``
    code : str, optional (default ``'vasp'``)
        The DFT code in which the calculation is performed with.
        Also, if you want to read a .bxsf file set code ="bxsf"
        e.g. ``code='vasp'``
    vmin : float, optional
        The maximum value in the color bar. cmap is only relevant in
        ``mode='parametric'``.
        e.g. vmin=-1.0
    vmax : float, optional
        The maximum value in the color bar. cmap is only relevant in
        ``mode='parametric'``.
        e.g. vmax=1.0
    savegif : str, optional
        pyprocar can save the fermi surface in a gif
        format. ``savegif`` is the path to which the gif is saved.
        e.g. ``savegif='fermi.gif'`` or ``savegif='~/MgB2/fermi.gif'``
    savemp4 : str, optional
        pyprocar can save the fermi surface in a mp4 video format.
        ``savemp4`` is the path to which the video is saved.
        e.g. ``savegif='fermi.mp4'`` or ``savegif='~/MgB2/fermi.mp4'``
    save3d : str, optional
        pyprocar can save the fermi surface in a 3d file format.
        pyprocar uses the `trimesh <https://github.com/mikedh/trimesh>`_
        to save the 3d file. trimesh can export files with the
        following formats STL, binary PLY, ASCII OFF, OBJ, GLTF/GLB
        2.0, COLLADA. ``save3d`` is the path to which the file is saved.
        e.g. ``save3d='fermi.glb'``
    save_meshio : bool, optional
        pyprocar can use meshio to save any 3d format supported by it.
    perspective : bool, optional
        To create the illusion of depth, perspective is used in 2d
        graphics. One can turn this feature off by ``perspective=False``
        e.g.  ``perspective=False``
    save2d : str, optional
        The fermi surface can be saved as a 2D image. This parameter
        turns this feature on and selects the path at which the file
        is going to be saved.
        e.g. ``save2d='fermi.png'``
    show_slice : bool, optional
        Creates a widget which slices the fermi surface
    slice_origin : tuple, optional
        Origin to put the plane widget
    slice_normal : bool, optional
        Normal of the plane widget
    show_cross_section_area : bool, optional
        Shows the largest cross sectional area
    show_curvature : bool, optional
        plots the curvature of the fermi surface
    curvature_type : str, optional
        If show_curvature is True, this option chooses the type of curvature 
        availible in Pyvista. ('mean', 'gaussian', 'maximum', 'minimum')
    iso_slider : bool, optional
        plots a slider widget which controls which iso_energy value viewed
    iso_range : float, optional
        If iso_slider is True, this specifies the energy range 
        around the fermi surface to view
    iso_surfaces : int, optional
        If iso_slider is True, this specifies how many surfaces to 
        generate in the range specified around the fermi surface
    camera_pos : list float, optional (default ``[1, 1, 1]``)
        This parameter defines the position of the camera where it is
        looking at the fermi surface. This feature is important when
        one chooses to use the ``save2d``, ``savegif`` or ``savemp4``
        option.
        e.g. ``camera_pos=[0.5, 1, -1]``
    widget : , optional
        .. todo::
    show : bool, optional (default ``True``)
        If set to ``False`` it will not show the 3D plot.
    Returns
    -------
    s : pyprocar surface object
        The whole fermi surface added bands
    surfaces : list pyprocar surface objects
        list of fermi surface of each band
    """

    welcome()
    ##########################################################################
    # Code dependencies
    ##########################################################################
    if code == "vasp" or code == "abinit":
        if repair:
            repairhandle = UtilsProcar()
            repairhandle.ProcarRepair(procar, procar)
            print("PROCAR repaired. Run with repair=False next time.")

    if code == "vasp":
        outcar = io.vasp.Outcar(filename=outcar)
            
        e_fermi = outcar.efermi
        
        poscar = io.vasp.Poscar(filename=poscar)
        structure = poscar.structure
        reciprocal_lattice = poscar.structure.reciprocal_lattice

        parser = io.vasp.Procar(filename=procar,
                                structure=structure,
                                reciprocal_lattice=reciprocal_lattice,
                                efermi=e_fermi,
                                )

    elif code == "abinit":
        parser = ProcarParser()
        parser.readFile(procar, False)
        abinitFile = AbinitParser(abinit_output=abinit_output)
        if fermi is None:
            e_fermi = abinitFile.fermi
        else:
            e_fermi = fermi
        reciprocal_lattice = abinitFile.reclat

        # Converting Ha to eV
        parser.bands = 27.211396641308 * parser.bands

    elif code == "qe":
        # procarFile = parser
        if dirname is None:
            dirname = "bands"
        parser = io.qe.QEParser(scfIn_filename = "scf.in", dirname = dirname, bandsIn_filename = "bands.in", 
                             pdosIn_filename = "pdos.in", kpdosIn_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
                             dos_interpolation_factor = None)
        reciprocal_lattice = parser.reciprocal_lattice
        if fermi is None:
            e_fermi = parser.efermi
        else:
            e_fermi = fermi

        # procarFile = QEFermiParser()
        # reciprocal_lattice = procarFile.reclat
        # data = ProcarSelect(procarFile, deepCopy=True)
        # if fermi is None:
        #     e_fermi = procarFile.efermi
        # else:
        #     e_fermi = fermi

    elif code == "lobster":
        parser = LobsterFermiParser()
        reciprocal_lattice = parser.reclat
        if fermi is None:
            e_fermi = 0
        else:
            e_fermi = fermi


    elif code == "bxsf":
        e_fermi = fermi
        parser = BxsfParser(infile= infile)
        reciprocal_lattice = parser.reclat

    elif code == "frmsf":
        e_fermi = fermi
        parser = FrmsfParser(infile=infile)
        reciprocal_lattice = parser.rec_lattice
        bands = np.arange(len(parser.bands[0, :]))

    parser.ebs.bands += e_fermi


    ##########################################################################
    # Data Formating
    ##########################################################################


    bands_to_keep = bands
    if bands_to_keep is None:
        bands_to_keep = np.arange(len(parser.bands[0, :]))


    spd = []
    
    if mode == "parametric":
        if orbitals is None:
            orbitals = np.arange(parser.ebs.norbitals, dtype=int)
        if atoms is None:
            atoms = np.arange(parser.ebs.natoms, dtype=int)
        if spins is None:
            spins = [0]

        # data.selectIspin(spins)
        # data.selectAtoms(atoms, fortran=False)
        # data.selectOrbital(orbitals)

        projected = parser.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        projected = projected[:,:,spins[0]]

        for iband in bands_to_keep:
            spd.append(projected[:,iband] )
    elif mode == "property_projection":
        for iband in bands_to_keep:
            spd.append(None)
    else:
        for iband in bands_to_keep:
            spd.append(None)
   
    spd_spin = []

    if spin_texture:
        ebsX = copy.deepcopy(parser.ebs)
        ebsY = copy.deepcopy(parser.ebs)
        ebsZ = copy.deepcopy(parser.ebs)

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
    else:
        for iband in bands_to_keep:
            spd_spin.append(None)



    ##########################################################################
    # Initialization of the Fermi Surface
    ##########################################################################
    if iso_slider == False:
        fermi_surface3D = FermiSurface3D(
                                        kpoints=parser.ebs.kpoints,
                                        bands=parser.ebs.kpoints,
                                        bands_to_keep = bands_to_keep,
                                        spd=spd,
                                        spd_spin=spd_spin,
                                        colors = colors,
                                        calculate_fermi_speed = calculate_fermi_speed,
                                        calculate_fermi_velocity = calculate_fermi_velocity,
                                        calculate_effective_mass = calculate_effective_mass,
                                        fermi=e_fermi,
                                        fermi_shift = fermi_shift,
                                        fermi_tolerance=fermi_tolerance,
                                        reciprocal_lattice=reciprocal_lattice,
                                        interpolation_factor=interpolation_factor,
                                        projection_accuracy=projection_accuracy,
                                        supercell=supercell,
                                        cmap=cmap,
                                        vmin = vmin,
                                        vmax=vmax,
                                        extended_zone_directions = extended_zone_directions,
                                        # curvature_type = curvature_type,
                                    )
        

        brillouin_zone = fermi_surface3D.brillouin_zone

        # fermi_surface_area = fermi_surface3D.fermi_surface_area
        # band_surfaces_area = fermi_surface3D.band_surfaces_area
        
        # fermi_surface_curvature = fermi_surface3D.fermi_surface_curvature
        # band_surfaces_curvature = fermi_surface3D.band_surfaces_curvature
    
        
    elif iso_slider == True:
       
        energy_values = np.linspace(e_fermi-iso_range/2,e_fermi+iso_range/2,iso_surfaces)
        e_surfaces = []
        
    
        for e_value in energy_values:
            fermi_surface3D = FermiSurface3D(
                                            kpoints=parser.ebs.kpoints,
                                            bands_to_keep = bands_to_keep,
                                            bands=parser.ebs.kpoints,
                                            spd=spd,
                                            spd_spin=spd_spin,
                                            calculate_fermi_speed = calculate_fermi_speed,
                                            calculate_fermi_velocity = calculate_fermi_velocity,
                                            calculate_effective_mass = calculate_effective_mass,
                                            fermi=e_value,
                                            fermi_shift = fermi_shift,
                                            fermi_tolerance=fermi_tolerance,
                                            reciprocal_lattice=reciprocal_lattice,
                                            interpolation_factor=interpolation_factor,
                                            projection_accuracy=projection_accuracy,
                                            supercell=supercell,
                                            cmap=cmap,
                                            vmin = vmin,
                                            vmax=vmax,
                                            extended_zone_directions = extended_zone_directions
                                        )
            brillouin_zone = fermi_surface3D.brillouin_zone
            e_surfaces.append(fermi_surface3D)
       
    
    ##########################################################################
    # Plotting the surface
    ##########################################################################
    if show:
        p = pyvista.Plotter()

    if show or save2d:

        # sargs = dict(interactive=True)


        p.add_mesh(
            brillouin_zone,
            style="wireframe",
            line_width=3.5,
            color="black",
        )
        
                
        if show_slice == True:
            text = mode
            if show_cross_section_area == True and bands != None:
                if len(bands) == 1:
                    add_mesh_slice_w_cross_sectional_area(plotter = p, mesh=fermi_surface3D, normal =slice_normal,origin = slice_origin, scalars = "scalars")
                    p.remove_scalar_bar()
                else:
                    print('---------------------------------------------')
                    print("Can only show area of one band at a time")
                    print('---------------------------------------------')
            else:
                if mode == "plain":
                    text = "Plain"
                    add_custom_mesh_slice(plotter = p, mesh=fermi_surface3D, normal =slice_normal,origin = slice_origin,  scalars = "scalars")
                    p.remove_scalar_bar()
                elif mode == "parametric":
                    text = "Projection"
                    add_custom_mesh_slice(plotter = p, mesh=fermi_surface3D, normal =slice_normal,origin = slice_origin,  scalars = "scalars")

                    p.remove_scalar_bar()
                    
        elif iso_slider == True:
            def create_mesh(value):
                res = int(value)
                closest_idx = find_nearest(energy_values, res)
                if mode == "plain":
                    scalars = "bands"
                elif mode == "parametric":
                    scalars = "scalars"
                elif mode == "property_projection":
                    if calculate_fermi_speed == True:
                        scalars = "Fermi Speed"
                    elif calculate_fermi_velocity == True:
                        scalars = "Fermi Velocity Vector"
                        
                    elif calculate_effective_mass == True:
                        scalars = "Geometric Average Effective Mass"
                else:
                    text = "Spin Texture"
                    scalars = "spin"
                    
                    
                p.add_mesh(e_surfaces[closest_idx], name='iso_surface', scalars = scalars)
                arrows =e_surfaces[closest_idx].glyph(
                orient=scalars,scale=False ,factor=arrow_size)
                p.remove_scalar_bar()
                
                if arrow_color is None:
                    p.add_mesh(arrows, scalars = "Fermi Velocity Vector_magnitude" ,cmap=cmap)
                else:
                    p.add_mesh(arrows, color=arrow_color)
                p.remove_scalar_bar()
                
                return
            if mode == "plain":
                text = "Plain"
            elif mode == "parametric":
                text = "Projection"
            elif mode == "property_projection":
                if calculate_fermi_speed == True:
                    text = "Projection"
            else:
                text = "Spin Texture"
                
                    
            p.add_slider_widget(create_mesh, [np.amin(energy_values), np.amax(energy_values)], title='Energy iso-value',style='modern',color = 'black')
            
            
            
        else:

            if not spin_texture:
                if mode == "plain":
                    p.add_mesh(fermi_surface3D, scalars = "bands",cmap = cmap, rgba = True)
                    text = "Plain"
                elif mode == "parametric":
                    p.add_mesh(fermi_surface3D, scalars = "scalars", cmap=cmap)
                    p.remove_scalar_bar()
                    text = "Projection"
                elif mode == "property_projection":
                    if calculate_fermi_speed == True:
                        text = "Fermi Speed"
                        p.add_mesh(fermi_surface3D,scalars = "Fermi Speed", cmap=cmap)
                    if calculate_effective_mass == True:
                        text = "Effective Mass"
                        p.add_mesh(fermi_surface3D,scalars =  "Geometric Average Effective Mass", cmap=cmap)
                        
                    if calculate_fermi_velocity == True:
                        text = "Fermi Velocity"

                        arrows = fermi_surface3D.glyph(
                        orient="Fermi Velocity Vector",scale=False ,factor=arrow_size)
                        p.add_mesh(fermi_surface3D, scalars = "Fermi Velocity Vector_magnitude" , cmap=cmap)
                        p.remove_scalar_bar()
                        if arrow_color is None:
                            p.add_mesh(arrows, scalars = "Fermi Velocity Vector_magnitude" ,cmap=cmap)
                        else:
                            p.add_mesh(arrows, color=arrow_color)
                    p.remove_scalar_bar()
            else:
                text = "Spin Texture"
                # Example dataset with normals
                # create a subset of arrows using the glyph filter
                arrows = fermi_surface3D.glyph(
                orient="spin",scale=False ,factor=arrow_size)

                if arrow_color is None:
                    p.add_mesh(arrows, cmap=cmap, clim=[vmin, vmax])
                    p.remove_scalar_bar()
                else:
                    p.add_mesh(arrows, color=arrow_color)

        
        if mode != "plain" or spin_texture:
            p.add_scalar_bar(
                title=text,
                n_labels=6,
                italic=False,
                bold=False,
                title_font_size=None,
                label_font_size=None,
                position_x=0.4,
                position_y=0.01,
                color="black",
            )

        p.add_axes(
            xlabel="Kx", ylabel="Ky", zlabel="Kz", line_width=6, labels_off=False
        )

        if not perspective:
            p.enable_parallel_projection()

        p.set_background(background_color)
        if not widget:
            p.show(cpos=camera_pos, screenshot=save2d)
        if savegif is not None:
            path = p.generate_orbital_path(n_points=36)
            p.open_gif(savegif)
            p.orbit_on_path(path) 
        if savemp4:
            path = p.generate_orbital_path(n_points=36)
            p.open_movie(savemp4)
            p.orbit_on_path(path) 
            # p.close()
    # p.show()
    # if iso_slider == False:
    #     s = boolean_add(fermi_surface3D)
    # s.set_color_with_cmap(cmap=cmap, vmin=vmin, vmax=vmax)
    # s.pyvista_obj.plot()
    # s.trimesh_obj.show()

    if save3d is not None:
        if save_meshio == True:
            pyvista.save_meshio(save3d,  fermi_surface3D)
        else:
            extention = save3d.split(".")[-1]
    #         s.export(save3d, extention)
    # if iso_slider == False:
    #     return s, fermi_surface3D
    

def add_mesh_slice_w_cross_sectional_area(plotter, mesh, normal='x', generate_triangles=False,
                                        widget_color=None, assign_to_axis=None,
                                        tubing=False, origin_translation=True,origin = (0,0,0),
                                        outline_translation=False, implicit=True,
                                        normal_rotation=True, **kwargs):

        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        plotter.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        # if not hasattr(self, "plane_sliced_meshes"):
        plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pyvista.wrap(alg.GetOutput())
        plotter.plane_sliced_meshes.append(plane_sliced_mesh)
        
        user_slice = plotter.plane_sliced_meshes[0]
        surface = user_slice.delaunay_2d()
        plotter.add_text(f'Cross sectional area : {surface.area}', color = 'black')
        def callback(normal, origin):
            # create the plane for clipping
            
            plane = generate_plane(normal, origin)
            
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            
            user_slice = plotter.plane_sliced_meshes[0]
            surface = user_slice.delaunay_2d()
            text = f'Cross sectional area : {surface.area}'
            plotter.textActor.SetText(2, text)


        plotter.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal='x',
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=origin,
                              normal_rotation=normal_rotation)
    
        actor = plotter.add_mesh(plane_sliced_mesh, **kwargs)
        
        plotter.plane_widgets[0].SetNormal(normal)

        return actor
    
def add_custom_mesh_slice(plotter, mesh, normal='x', generate_triangles=False,
                       widget_color=None, assign_to_axis=None,
                       tubing=False, origin_translation=True,origin = (0,0,0),
                       outline_translation=False, implicit=True,
                       normal_rotation=True, **kwargs):

        name = kwargs.get('name', mesh.memory_address)
        rng = mesh.get_data_range(kwargs.get('scalars', None))
        kwargs.setdefault('clim', kwargs.pop('rng', rng))
        mesh.set_active_scalars(kwargs.get('scalars', mesh.active_scalars_name))

        plotter.add_mesh(mesh.outline(), name=name+"outline", opacity=0.0)

        alg = vtk.vtkCutter() # Construct the cutter object
        alg.SetInputDataObject(mesh) # Use the grid as the data we desire to cut
        if not generate_triangles:
            alg.GenerateTrianglesOff()

        # if not hasattr(self, "plane_sliced_meshes"):
        plotter.plane_sliced_meshes = []
        plane_sliced_mesh = pyvista.wrap(alg.GetOutput())
        plotter.plane_sliced_meshes.append(plane_sliced_mesh)
        

        def callback(normal, origin):
            # create the plane for clipping
            
            plane = generate_plane(normal, origin)
            
            alg.SetCutFunction(plane) # the cutter to use the plane we made
            alg.Update() # Perform the Cut
            
            plane_sliced_mesh.shallow_copy(alg.GetOutput())
            



        plotter.add_plane_widget(callback=callback, bounds=mesh.bounds,
                              factor=1.25, normal='x',
                              color=widget_color, tubing=tubing,
                              assign_to_axis=assign_to_axis,
                              origin_translation=origin_translation,
                              outline_translation=outline_translation,
                              implicit=implicit, origin=origin,
                              normal_rotation=normal_rotation)
    
        actor = plotter.add_mesh(plane_sliced_mesh, **kwargs)
        
        plotter.plane_widgets[0].SetNormal(normal)

        return actor


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

