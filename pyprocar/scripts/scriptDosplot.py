__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .scriptDosplot_old import dosplot_old
from ..splash import welcome
from .. import io
from ..utils.info import orbital_names
from ..plotter import DOSPlot
from ..utils.defaults import settings


def dosplot(
        filename:str="vasprun.xml",
        dirname:str=None,
        poscar:str='POSCAR',
        procar:str="PROCAR",
        outcar:str='OUTCAR',
        mode:str="plain",
        interpolation_factor:int=1,
        orientation:str="horizontal",
        spin_colors:List[str] or List[Tuple[int,int,int]]=None,
        spin_labels:List[str]=None,
        colors:List[str] or List[Tuple[int,int,int]]=None,
        spins:List[int]=None,
        atoms:List[int]=None,
        orbitals:List[int]=None,
        items:dict={},
        fermi:float=None,
        elimit:List[float]=None,
        dos_limit:List[float]=None,
        cmap:str="jet",
        linewidth:float=1,
        vmax:float=None,
        vmin:float=None,
        grid:bool=False,
        savefig:str=None,
        title:str=None,
        plot_total:bool=True,
        projection_mask=None,
        code:str="vasp",
        lobster:bool=False,
        labels:List[str]=None, 
        ax:plt.Axes=None,
        verbose:bool=True,
        old:bool=False,
        show:bool=True
):
    """
    This function plots the density of states in different formats

    Parameters
    ----------

    filename : str, optional (default ``'vasprun.xml'``)
        The most important argument needed dosplot is
        **filename**. **filename** defines the path to `vasprun.xml`
        from the density of states calculation. If plotting is being
        carried out in the directory of the calculation, one does not
        need to specify this argument.

        e.g. ``filename='~/SrVO3/DOS/vasprun.xml'``

    dirname : str, optional (default ``'vasprun.xml'``)
        This is used for qe and lobster codes. It specifies the directory the dosplot
        calculation was performed.
        
        e.g. ``dirname='~/SrVO3/dos'``

    mode : str, optional (default ``'plain'``)
        **mode** defines the mode of the plot. This parameter will be
        explained in details with exmaples in the tutorial.
        options are ``'plain'``, ``'parametric'``,
        ``'parametric_line'``, ``'stack'``,
        ``'stack_orbitals'``, ``'stack_species'``.

        e.g. ``mode='stack'``

    interpolation_factor : int, optional (default ``None``)
        Number of points in energy axis is multiplied by this factor
        and interpolated using cubic
        spline.

        e.g. ``interpolation_factor=3``

    orientation : str, optional (default ``horizontal'``)
        The orientation of the DOS plot.  options are
        ``'horizontal', 'vertical'``

        e.g. ``orientation='vertical'``

    spin_colors : list str or tuples, (optional ``spin_colors=['blue',
        'red']``)
        **spin_colors** represent the colors the different spin
        ploarizations are going to be represented in the DOS
        plot. These colors can be chosen from any type of color
        acceptable by matplotlib(string,rgb,html).

        e.g. ``spin_colors=['blue','red']``,
        ``spin_colors=[(0, 0, 1),(1, 0,0 )]``,
        ``spin_colors=['#0000ff','#ff0000']``

        .. caution:: If the calculation is spin polarized one has to
        provide two colors even if one is plotting one spin. I
        disregard this cation if using default.

    colors : list str or tuples, optional (default, optional)
        ``colors`` defines the color of plots filling the area under
        the curve of Total density of states. This is only important in the
        ``mode=stack``, ``mode=stack_species``,
        ``mode=stack_orbitals``. To have a better sense of this
        parameter refer to the stack plots of  SrVO\ :sub:`3`\. These
        colors can be chosen from any type of color acceptable by
        matplotlib(string,rgb,html).

        e.g. ``colors=['red', 'blue', 'green', 'magenta', 'cyan']``

    spins : list int, optional
        ``spins`` defines plotting of different spins channels present
        in the calculation, If the calculation is spin non-polorized
        the spins will be set by default to ``spins=[0]``. if the
        calculation is spin polorized this parameter can be set to 0
        or 1 or both.

        e.g. ``spins=[0, 1]``

    spin_labels : list str, optional
        ``spin_labels`` defines labels to use to represent spin 
        in the legend of the plot.
        e.g. ``spin_labels=['$\uparrow$','$\downarrow$']``

    atoms : list int, optional
        ``atoms`` define the projection of the atoms in the Density of
        States. In other words it selects only the contribution of the
        atoms provided. Atoms has to be a python list(or numpy array)
        containing the atom indices. Atom indices has to be order of
        the input files of DFT package. ``atoms`` is only relevant in
        ``mode='parametric'``, ``mode='parametric_line'``,
        ``mode='stack_orbitals'``. keep in mind that python counting
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
        ``orbitals`` define the projection of orbitals in the density
        of States. In other words it selects only the contribution of
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
        ``orbitals`` is only relavent in ``mode='parametric'``,
        ``mode='parametric_line'``, ``mode='stack_species'``.

        e.g. ``orbitals=[1,2,3]`` will only select the p orbitals
        while ``orbitals=[4,5,6,7,8]`` will select the d orbitals.

        If nothing is specified pyprocar will select all the present
        orbitals.

    elimit : list float, optional
        Energy window limit asked to plot. ``elimit`` has to be a two
        element python list(or numpy array).

        e.g. ``elimit=[-2, 2]``
        The default is set to the minimum and maximum of the energy
        window.

    dos_limit : list float, optional
       ``dos_limit`` defines the density of states axis limits on the
       graph. It is automatically set to select 10% higher than the
       maximum of density of states in the specified energy window.

       e.g. ``dos_limit=[0, 30]``

    cmap : str , optional (default 'jet')
        The color map used for color coding the projections. ``cmap``
        is only relevant in ``mode='parametric'``. a full list of
        color maps in matplotlib are provided in this web
        page. `https://matplotlib.org/2.0.1/users/colormaps.html
        <https://matplotlib.org/2.0.1/users/colormaps.html>`_

        e.g. ``cmap='plasma'``

    linewidth : float, optional (default 1)
        The line width with which the total DOS is ploted

        e.g. linewidth=2

    vmax : float, optional
        The maximum value in the color bar. ``cmap`` is only relevant
        in ``mode='parametric'``.

        e.g. ``vmax=1.0``

    vmin : float, optional
        The maximum value in the color bar. ``cmap`` is only relevant
        in ``mode='parametric'``.

        e.g. ``vmin=-1.0``

    grid : bool, optional (default Flase)
        Defines If a grid is plotted in the plot. The entry should be
        python boolian.

        e.g. ``grid=True``

    savefig : str , optional (default None)
        ``savefig`` defines the file that the plot is going to be
        saved in. ``savefig`` accepts all the formats accepted by
        matplotlib such as png, pdf, jpg, ...
        If not provided the plot will be shown in the
        interactive matplotlib mode.

        e.g. ``savefig='DOS.png'``, ``savefig='DOS.pdf'``

    title : str, optional
        Defines the plot title asked to be added above the plot. If
        ``title`` is not defined, PyProcar will not add any title.

        e.g. ``title="Total Density of States SrVO_$3$"``. One can use
        LaTex format as well.

    plot_total : bool, optional (default ``True``)
        If the total density of states is plotted as well as other
        options. The entry should be python boolian.

        e.g. ``plot_total=True``

    code : str, optional (default ``'vasp'``)
        Defines the Density Functional Theory code used for the
        calculation. The default of this argument is vasp, so if the
        cal is done in vasp one does not need to define this argumnet.

        e.g. ``code=vasp``, ``code=elk``, ``code=abinit``

    labels : list str, optional
        ``labels`` define the legends plotted in defining each spin.

        e.g.  ``labels=['Oxygen-Up','Oxygen-Down']``,
        ``labels=['Oxygen-'+r'$\\uparrow$','Oxygen-'+r'$\\downarrow$']``
        Side means the string will be treated as raw string. This has
        to be used if LaTex formating is used.
        No default is used in the ``mode=plain``, ``mode=parametric``,
        ``mode=parametric_line``. In ``mode=stack``, `ack_species``,
        ``mode=stack_orbitals`` the labels are generated automatically
        based on the other parameters such as atoms and orbitals.

    items : dict, optional
        ``items`` is only relavent for ``mode='stack'``. stack will
        plot the items defined with stacked filled areas under
        curve. For clarification visit the examples in the
        tutorial. ``items`` need to be provided as a python
        dictionary, with keys being specific species and values being
        projections of ``orbitals``. The following examples can
        clarify the python lingo.

        e.g.  ``items={'Sr':[0],'O':[1,2,3],'V':[4,5,6,7,8]}`` or
        ``items=dict(Sr=[0],O=[1,2,3],V=[4,5,6,7,8])``. The two
        examples are equivalent to each other. This will plot the
        following curves stacked on top of each other. projection of s
        orbital in Sr, projection of p orbitals in O and projection of
        d orbitals in V.
        The default is set to take every atom and every orbital. Which
        will be equivalent to ``mode='stack_species'``

    ax : matplotlib ax object, optional
        ``ax`` is a matplotlib axes. In case one wants to put plot
        generated from this plot in a different figure and treat the
        output as a subplot in a larger plot.

        e.g. ::

            >>> # Creates a figure with 3 rows and 2 colomuns
            >>> fig, axs = plt.subplots(3, 2)
            >>> x = np.linspace(-np.pi, np.pi, 1000)
            >>> y = np.sin(x)
            >>> axs[0, 0].plot(x, y)
            >>> pyprocar.dosplot(mode='plain',ax=axs[2, 2]),elimit=[-2,2])
            >>> plt.show()

    plt_show : bool, optional (default ``True``)
        whether to show the generated plot or skip to the saving.

        e.g. ``plt_show=True``


    Returns
    -------
    fig : matplotlib figure
        The generated figure

    ax : matplotlib ax object
        The generated ax for this density of states.
        If one chooses ``plt_show=False``, one can modify the plot
        using this returned object.
        e.g. ::

            >>> fig, ax = pyprocar.dosplot(mode='plain', plt_show=False)
            >>> ax.set_ylim(-2,2)
            >>> fig.show()

    """
    
    
    if mode not in [
            'plain', 'parametric_line', 'parametric', 'stack_species',
            'stack_orbitals', 'stack']:
        raise ValueError(
            "Mode should be choosed from ['plain', 'parametric_line','parametric','stack_species','stack_orbitals','stack']"
        )

    if orientation[0].lower() == 'h':
        orientation = 'horizontal'
    elif orientation[0].lower() == 'v':
        orientation = 'vertical'


    dos, structure, reciprocal_lattice = parse(
                                        code=code,
                                        filename=filename,
                                        lobster=lobster, 
                                        dirname=dirname, 
                                        outcar=outcar, 
                                        poscar=poscar, 
                                        procar=procar,
                                        interpolation_factor=interpolation_factor, 
                                        fermi=fermi)

    if elimit is None:
        elimit = [dos.energies.min(), dos.energies.max()]
    
    edos_plot = DOSPlot(dos = dos, structure = structure, spins = spins, orientation = orientation)
    
    if mode == "plain":
        edos_plot.plot_dos(orientation = orientation)

    if mode == "parametric":
        if atoms is None:
            atoms = list(np.arange(edos_plot.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(edos_plot.dos.total)))
        if orbitals is None:
            orbitals = list(np.arange(len(edos_plot.dos.projected[0][0]), dtype=int))
        
        edos_plot.plot_parametric(
                        atoms=atoms,
                        principal_q_numbers=[-1],
                        orbitals=orbitals,
                        spin_colors=spin_colors,
                        spin_labels=spin_labels,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        orientation = orientation,
                        plot_total=plot_total,
                        plot_bar=True)

    if mode == "parametric_line":
        if atoms is None:
            atoms = list(np.arange(edos_plot.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(edos_plot.dos.total)))
        if orbitals is None:
            orbitals = list(np.arange(len(edos_plot.dos.projected[0][0]), dtype=int))
        
        edos_plot.plot_parametric_line(
                        atoms=atoms,
                        principal_q_numbers=[-1],
                        orbitals=orbitals,
                        spin_colors=spin_colors,
                        orientation=orientation
                        )

    if mode == "stack_species":
        edos_plot.plot_stack_species(
            orbitals=orbitals,
            spin_colors=spin_colors,
            spin_labels = spin_labels,
            colors = colors,
            plot_total = plot_total,
            orientation=orientation,
        )

    elif mode == "stack_orbitals":
        edos_plot.plot_stack_orbitals(
            atoms=atoms,
            spin_colors=spin_colors,
            spin_labels = spin_labels,
            colors = colors,
            plot_total = plot_total,
            orientation=orientation,
        )

    elif mode == "stack":
        edos_plot.plot_stack(
            items=items,
            spin_colors=spin_colors,
            spin_labels = spin_labels,
            colors=colors,
            orientation=orientation,
            plot_total = plot_total,
        )

    edos_plot.draw_fermi(
            orientation = orientation,
            color=settings.edos.fermi_color,
            linestyle=settings.edos.fermi_linestyle,
            linewidth=settings.edos.fermi_linewidth,
        )
    if orientation == 'horizontal':
        if elimit is not None:
            edos_plot.set_xlim(elimit)
        if dos_limit is not None:
            edos_plot.set_ylim(dos_limit)
    elif orientation == 'vertical' :
        if elimit is not None:
            edos_plot.set_ylim(elimit)
        if dos_limit is not None:
            edos_plot.set_xlim(dos_limit)

    if settings.edos.grid or grid:
        edos_plot.grid()
    if settings.edos.legend and len(edos_plot.labels) != 0:
        edos_plot.legend(edos_plot.labels)
    if savefig is not None:
        edos_plot.save(savefig)
    if show:
        edos_plot.show()
    return edos_plot

def parse(code: str='vasp',
          filename:str='vasprun.xml',
          lobster: bool=False,
          dirname: str="",
          outcar:str='OUTCAR',
          poscar:str='POSCAR',
          procar:str='PROCAR',
          interpolation_factor:int=1,
          fermi:float=None):
    ebs = None
    kpath = None
    structure = None


    if lobster is True:
        if dirname is None:
            dirname = "dos"
        parser = io.lobster.LobsterParser(dirname = dirname,code = code, dos_interpolation_factor = None)
        if fermi is None:
            fermi = parser.efermi
        
        reciprocal_lattice = parser.reciprocal_lattice
    
        structure = parser.structure
        
        dos = parser.dos

    elif code == "vasp":
        if outcar is not None:
            outcar = io.vasp.Outcar(outcar)
            if fermi is None:
                fermi = outcar.efermi
            reciprocal_lattice = outcar.reciprocal_lattice
        if poscar is not None:
            poscar = io.vasp.Poscar(poscar)
            structure = poscar.structure
            if reciprocal_lattice is None:
                reciprocal_lattice = poscar.structure.reciprocal_lattice


        vaspxml = io.vasp.VaspXML(filename=filename,
                               dos_interpolation_factor=None) 
        
        dos = vaspxml.dos
        
        
    elif code == "qe":
        if dirname is None:
            dirname = "dos"
        parser = io.qe.QEParser(scfIn_filename = "scf.in", dirname = dirname, bandsIn_filename = "bands.in", 
                             pdosIn_filename = "pdos.in", kpdosIn_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
                             dos_interpolation_factor = None)
        if fermi is None:
            fermi = parser.efermi
        
        reciprocal_lattice = parser.reciprocal_lattice
    
        structure = parser.structure
        
        dos = parser.dos
        
    
    return dos,  structure, reciprocal_lattice

