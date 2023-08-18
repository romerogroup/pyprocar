# -*- coding: utf-8 -*-
__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import os
from typing import List, Tuple
import yaml

import numpy as np
import matplotlib.pyplot as plt

from ..utils import welcome, ROOT
from .. import io
from ..utils.info import orbital_names
from ..plotter import DOSPlot

with open(os.path.join(ROOT,'pyprocar','cfg','dos.yml'), 'r') as file:
    plot_opts = yaml.safe_load(file)

def dosplot(
        code:str="vasp",
        dirname:str=None,
        mode:str="plain",
        interpolation_factor:int=1,
        orientation:str="horizontal",
        spins:List[int]=None,
        atoms:List[int]=None,
        orbitals:List[int]=None,
        items:dict={},
        fermi:float=None,
        elimit:List[float]=None,
        dos_limit:List[float]=None,
        savefig:str=None,
        projection_mask=None,
        ax:plt.Axes=None,
        verbose:bool=True,
        show:bool=True,
        print_plot_opts:bool=False,
        **kwargs
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

    spins : list int, optional
        ``spins`` defines plotting of different spins channels present
        in the calculation, If the calculation is spin non-polorized
        the spins will be set by default to ``spins=[0]``. if the
        calculation is spin polorized this parameter can be set to 0
        or 1 or both.

        e.g. ``spins=[0, 1]``

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

        .. code-block::
            :linenos:
            
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

    savefig : str , optional (default None)
        ``savefig`` defines the file that the plot is going to be
        saved in. ``savefig`` accepts all the formats accepted by
        matplotlib such as png, pdf, jpg, ...
        If not provided the plot will be shown in the
        interactive matplotlib mode.

        e.g. ``savefig='DOS.png'``, ``savefig='DOS.pdf'``

    plot_total : bool, optional (default ``True``)
        If the total density of states is plotted as well as other
        options. The entry should be python boolian.

        e.g. ``plot_total=True``

    code : str, optional (default ``'vasp'``)
        Defines the Density Functional Theory code used for the
        calculation. The default of this argument is vasp, so if the
        cal is done in vasp one does not need to define this argumnet.

        e.g. ``code=vasp``, ``code=elk``, ``code=abinit``

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

    print_plot_opts: bool, optional
        Boolean to print the plotting options


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
    
    modes=['plain','parametric',
           'parametric_line', 'stack', 
           'stack_orbitals', 'stack_species']
    modes_txt=' , '.join(modes)
    message=f"""
            --------------------------------------------------------
            There are additional plot options that are defined in a configuration file. 
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}
            --------------------------------------------------------
            """
    print(message)
    if print_plot_opts:
        for key,value in plot_opts.items():
            print(key,':',value)
    

    if orientation[0].lower() == 'h':
        orientation = 'horizontal'
    elif orientation[0].lower() == 'v':
        orientation = 'vertical'

    
    parser = io.Parser(code = code, dir = dirname)
    dos = parser.dos
    structure = parser.structure

    if elimit is None:
        elimit = [dos.energies.min(), dos.energies.max()]
    
    edos_plot = DOSPlot(dos = dos, structure = structure, **kwargs)
    
    if mode == "plain":
        edos_plot.plot_dos(spins=spins, orientation = orientation)

    elif mode == "parametric":
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
                        spins=spins,
                        orientation=orientation)

    elif mode == "parametric_line":
        if atoms is None:
            atoms = list(np.arange(edos_plot.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(edos_plot.dos.total)))
        if orbitals is None:
            orbitals = list(np.arange(len(edos_plot.dos.projected[0][0]), dtype=int))
        
        edos_plot.plot_parametric_line(
                        atoms=atoms,
                        principal_q_numbers=[-1],
                        spins=spins,
                        orbitals=orbitals,
                        orientation=orientation
                        )

    elif mode == "stack_species":
        edos_plot.plot_stack_species(
            spins=spins,
            orbitals=orbitals,
            orientation=orientation,
        )

    elif mode == "stack_orbitals":
        edos_plot.plot_stack_orbitals(
            spins=spins,
            atoms=atoms,
            orientation=orientation,
        )

    elif mode == "stack":
        edos_plot.plot_stack(
            spins=spins,
            items=items,
            orientation=orientation,
        )
    else:
        raise ValueError("The mode needs to be in the List [plain,parametric,parametric_line,stack_species,stack_orbitals,stack]")

    edos_plot.draw_fermi(orientation = orientation)

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

    edos_plot.grid()
    edos_plot.legend(edos_plot.labels)
    if savefig is not None:
        edos_plot.save(savefig)
    if show:
        edos_plot.show()
    return edos_plot
