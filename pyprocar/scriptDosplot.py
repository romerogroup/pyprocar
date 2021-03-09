"""
Created on May 17 2020
@author: Pedram Tavadze
"""
# from .elkparser import ElkParser
from .splash import welcome
from .doscarplot import DosPlot
from .vaspxml import VaspXML
from .lobsterparser import LobsterDOSParser
from .qeparser import QEDOSParser

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
plt.rcParams["mathtext.default"] = "regular"
# Roman ['rm', 'cal', 'it', 'tt', 'sf',
#        'bf', 'default', 'bb', 'frak',
#        'circled', 'scr', 'regular']
plt.rcParams["font.family"] = "Arial"
plt.rc("font", size=18)  # controls default text sizes
plt.rc("axes", titlesize=22)  # fontsize of the axes title
plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=22)  # fontsize of the tick labels
plt.rc("ytick", labelsize=22)  # fontsize of the tick labels
# plt.rc('legend', fontsize=22)    # legend fontsize
# plt.rc('figure', titlesize=22)  # fontsize of the figure title


def dosplot(
        filename="vasprun.xml",
        mode="plain",
        interpolation_factor=None,
        orientation="horizontal",
        spin_colors=None,
        colors=None,
        spins=None,
        atoms=None,
        orbitals=None,
        elimit=None,
        dos_limit=None,
        cmap="jet",
        linewidth=1,
        vmax=None,
        vmin=None,
        grid=False,
        savefig=None,
        title=None,
        plot_total=True,
        code="vasp",
        labels=None,
        items={},
        ax=None,
        plt_show=True,
        verbose=True,
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

    linewidth : str, optional (default 1)
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
            'stack_orbitals', 'stack'
    ]:
        raise ValueError(
            "Mode should be choosed from ['plain', 'parametric_line','parametric','stack_species','stack_orbitals','stack']"
        )
    if verbose:
        welcome()

        # Verbose section
        print("Script initiated")
        print("code          : ", code)
        print("File name     : ", filename)
        print("mode          : ", mode)
        print("spins         : ", spins)
        print("atoms list    : ", atoms)
        print("orbs. list    : ", orbitals)
        print("energy range  : ", elimit)
        print("colormap      : ", cmap)
        print("vmax          : ", vmax)
        print("vmin          : ", vmin)
        print("grid enabled  : ", grid)
        print("savefig       : ", savefig)
        print("title         : ", title)

    total = plot_total
    code = code.lower()

    if orientation[0].lower() == 'h':
        orientation = 'horizontal'
    elif orientation[0].lower() == 'v':
        orientation = 'vertical'

    if code == "vasp":
        vaspxml = VaspXML(filename=filename,
                          dos_interpolation_factor=interpolation_factor)
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)
        if atoms is None:
            atoms = list(np.arange(vaspxml.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(vaspxml.dos.total)))
        if orbitals is None:
            orbitals = list(
                np.arange(len(vaspxml.dos.projected[0][0]), dtype=int))
        if elimit is None:
            elimit = [vaspxml.dos.energies.min(), vaspxml.dos.energies.max()]
    elif code == "lobster":
        vaspxml = LobsterDOSParser(
            filename="DOSCAR.lobster",
            dos_interpolation_factor=interpolation_factor)
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)
        if atoms is None:
            atoms = list(np.arange(vaspxml.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(vaspxml.dos.total)))
        if orbitals is None:
            orbitals = list(
                np.arange(len(vaspxml.dos.projected[0][0]), dtype=int))
        if elimit is None:
            elimit = [vaspxml.dos.energies.min(), vaspxml.dos.energies.max()]

    elif code == "qe":
        vaspxml = QEDOSParser(nscfin="nscf.in",
                              pdosin="pdos.in",
                              scfOut="scf.out",
                              dos_interpolation_factor=interpolation_factor)
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)
        if atoms is None:
            atoms = list(np.arange(vaspxml.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(vaspxml.dos.total)))
        if orbitals is None:
            orbitals = list(
                np.arange(len(vaspxml.dos.projected[0][0]), dtype=int))
        if elimit is None:
            elimit = [vaspxml.dos.energies.min(), vaspxml.dos.energies.max()]

    if mode == "plain":
        fig, ax1 = dos_plot.plot_total(
            spins=spins,
            spin_colors=spin_colors,
            ax=ax,
            orientation=orientation,
            labels=labels,
            linewidth=linewidth,
        )

    elif mode == "parametric_line":
        if not total:
            fig, ax1 = dos_plot.plot_parametric_line(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                ax=ax,
                orientation=orientation,
                labels=labels,
                linewidth=linewidth,
            )

        else:
            fig, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax,
                orientation=orientation,
                linewidth=linewidth,

            )
            _, ax1 = dos_plot.plot_parametric_line(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                ax=ax1,
                orientation=orientation,
                labels=labels,
                linewidth=linewidth,
            )
    elif mode == "parametric":
        if not total:
            fig, ax1 = dos_plot.plot_parametric(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                elimit=elimit,
                ax=ax,
                orientation=orientation,
                labels=labels,
            )
        else:
            fig, ax1 = dos_plot.plot_parametric(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                elimit=elimit,
                ax=ax,
                orientation=orientation,
                labels=labels,
            )

            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
                linewidth=linewidth,
            )

    elif mode == "stack_species":
        if not total:
            fig, ax1 = dos_plot.plot_stack_species(
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )
        else:
            fig, ax1 = dos_plot.plot_stack_species(
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )
            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
                linewidth=linewidth,
            )

    elif mode == "stack_orbitals":
        if not total:
            fig, ax1 = dos_plot.plot_stack_orbitals(
                spins=spins,
                atoms=atoms,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

        else:
            fig, ax1 = dos_plot.plot_stack_orbitals(
                spins=spins,
                atoms=atoms,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
                linewidth=linewidth,
            )

    elif mode == "stack":
        if not total:
            fig, ax1 = dos_plot.plot_stack(
                items=items,
                spins=spins,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

        else:
            fig, ax1 = dos_plot.plot_stack(
                items=items,
                spins=spins,
                spin_colors=colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
                linewidth=linewidth,
            )

    cond1 = vaspxml.dos.energies >= elimit[0]
    cond2 = vaspxml.dos.energies <= elimit[1]
    cond = np.all([cond1, cond2], axis=0)

    if dos_limit is not None:
        ylim = dos_limit
    else:
        if len(spins) > 1:
            ylim = [
                vaspxml.dos.total[1, cond].max() * -1.1,
                vaspxml.dos.total[0, cond].max() * 1.1
            ]
        else:
            ylim = [0, vaspxml.dos.total[spins[0], cond].max() * 1.1]

    if orientation == "horizontal":
        ax1.set_xlabel(r"$E-E_f$ [eV]")
        ax1.set_ylabel("Density of States [a.u.]")
        ax1.set_xlim(elimit)
        ax1.set_ylim(ylim)

    elif orientation == "vertical":
        ax1.set_ylabel(r"$E-E_f$ [eV]")
        ax1.set_xlabel("Density of States [a.u.]")
        ax1.set_ylim(elimit)
        ax1.set_xlim(ylim)  # we use ylim because the plot is vertical

    ax1.axhline(color="black", linestyle="--")
    ax1.axvline(color="black", linestyle="--")

    # fig.tight_layout()
    if grid:
        ax1.grid()
    if labels or "stack" in mode:
        ax1.legend()
    if title:
        ax1.set_title(title, fontsize=17)

    if savefig:

        fig.savefig(savefig, bbox_inches="tight")
        plt.close(
        )  # Added by Nicholas Pike to close memory issue of looping and creating many figures
        return None, None
    else:
        plt.show(block=plt_show)

    return fig, ax1


#
#
## if __name__ == "__main__":
## bandsplot(mode='parametric',elimit=[-6,6],orbitals=[4,5,6,7,8],vmin=0,vmax=1, code='elk')
## knames=['$\Gamma$', '$X$', '$M$', '$\Gamma$', '$R$','$X$'],
## kticks=[0, 8, 16, 24, 38, 49])
