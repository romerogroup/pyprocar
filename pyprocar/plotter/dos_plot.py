__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from ..utils.defaults import settings
from ..core import Structure, DensityOfStates

np.seterr(divide="ignore", invalid="ignore")

# TODO add python typing to all of the functions
# TODO Generalize orientation to remove if statments

class DOSPlot:
    """
    Class to plot an electronic band structure.

    Parameters
    ----------
    dos : DensityOfStates
        An density of states pyprocar.core.DensityOfStates.
    structure : Structure
        An density of states pyprocar.core.Structure.
    
    ax : mpl.axes.Axes, optional
        A matplotlib Axes object. If provided the plot will be located at that ax.
        The default is None.

    Returns
    -------
    None.

    """
    def __init__(self, 
                    dos:DensityOfStates=None, 
                    structure:Structure=None, 
                    ax:mpl.axes.Axes=None, 
                    **kwargs):
        settings.modify(kwargs)

        self.dos = dos
        self.structure = structure
        
        # if spins is None:
        #     self.spins = np.arange(self.dos.n_spins, dtype=int)
        #     self.nspins = len(self.spins)
        # else:
        #     self.spins = spins
        #     self.nspins = len(self.spins)


        self.handles = []
        self.labels = []
        if ax is None:
            self.fig = plt.figure(figsize=tuple(settings.general.figure_size),)
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax
    
        return None


    def plot_dos(self,
                spins:List[int]=None, 
                orientation:str = 'horizontal'):
        """
        Plot the plain density of states.

        Parameters
        ----------
        spins : list of ints, optional
            A list of the spins to be plotted. The default is None.
        color : string, optional
            Color for the bands. The default is "blue".

        Returns
        -------
        None.

        """
        if spins is None:
            if self.dos.is_non_collinear:
                spins = [0,1,2]
            else:
                spins = range(self.dos.n_spins)
        if self.dos.is_non_collinear:
            spins = [0]

        # plots over the different dos energies for spin polarized
        for ispin in spins:
            if orientation == 'horizontal':
                self.set_xlabel('Energy (eV)')
                self.set_ylabel('DOS')
                self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
                self.set_ylim([self.dos.total[ispin,:].min(),self.dos.total[ispin,:].max()])

                handle = self.ax.plot(
                    self.dos.energies, self.dos.total[ispin, :], color=settings.dos.spin_colors[ispin], alpha=settings.dos.opacity[
                        ispin], linestyle=settings.dos.linestyle[ispin], label=settings.dos.spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                )
            elif orientation == 'vertical':
                self.set_xlabel('DOS')
                self.set_ylabel('Energy (eV)')
                self.set_xlim([self.dos.total[ispin,:].min(),self.dos.total[ispin,:].max()])
                self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])
                handle = self.ax.plot(
                        self.dos.total[ispin, :], self.dos.energies, color=settings.dos.spin_colors[ispin], alpha=settings.dos.opacity[
                        ispin], linestyle=settings.dos.linestyle[ispin], label= settings.dos.spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                )
            self.handles.append(handle)

    def plot_parametric(self,
                        atoms:List[int]=None,
                        orbitals:List[int]=None,
                        spins:List[int]=None,
                        principal_q_numbers:List[int]=[-1],
                        spin_colors:List[str] or List[List[float]] =None,
                        spin_labels:List[str]=None,
                        cmap:str="jet",
                        vmin:float=0,
                        vmax:float=1,
                        plot_total:bool=True,
                        plot_bar:bool=True,
                        orientation:str='horizontal'):
        """The method will plot the parametric density of states

        Parameters
        ----------
        atoms : List[int], optional
            A list of atoms, by default None
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        principal_q_numbers : List[int], optional
            A list of principal quantum numbers, by default [-1]
        spin_colors : List[str] or List[List[float]], optional
            List of spin colors, by default None
        spin_labels : List[str], optional
            A list of spin labels, by default None
        cmap : str, optional
            The color map to use, by default "jet"
        vmin : float, optional
            Value to normalize the minimum projection value., by default 0
        vmax : float, optional
            Value to normalize the maximum projection value., by default 1
        plot_total : bool, optional
            Boolean to plot the total dos, by default True
        plot_bar : bool, optional
            Boolean to include colorbas, by default True
        orientation : str, optional
            String to plot horizontal or vertical plot, by default 'horizontal'
        """
        if spins is None:
            if self.dos.is_non_collinear:
                spins = [0,1,2]
            else:
                spins = range(self.dos.n_spins)
        spin_projections = spins

        # This covers the non-colinear case when spins only represent projections.  
        if self.dos.is_non_collinear:
            spins = [0]

        dos_total = np.array(self.dos.total)
        dos_total_projected = self.dos.dos_sum()
        dos_projected = self.dos.dos_sum(atoms=atoms,
                               principal_q_numbers=principal_q_numbers,
                               orbitals=orbitals,
                               spins=spin_projections)
        # print(np.where(np.logical_and( self.dos.energies>-51, self.dos.energies<-50)))
        if spin_colors is None:
            spin_colors = settings.dos.spin_colors
        if spin_labels is None:
            spin_labels = settings.dos.spin_labels

        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = (dos_projected.max() / dos_total_projected.max())

        cmap = mpl.cm.get_cmap(cmap)
        if plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax)


        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])

            if len(spins) == 2:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
            else:
                self.set_ylim([0,self.dos.total.max()])


            for spins_index , ispin in enumerate(spins):
                x = []
                y_total = []
                bar_color = []
                for idos in range(len(self.dos.energies)):
                    x.append(self.dos.energies[idos])
                    y = dos_projected[ispin][idos]
                    y_total.append(dos_total[ispin][idos])
                    y_total_projected = dos_total_projected[ispin][idos]
                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        y_total[-1] *= -1
                        y_total_projected *= -1

                    bar_color.append(cmap(y / (y_total_projected )))#* (vmax - vmin))))

                for idos in range(len(x) - 1):
                    self.ax.fill_between([x[idos], x[idos + 1]],
                                    [y_total[idos], y_total[idos + 1]],
                                    color=bar_color[idos])
                
                if plot_total == True:
                    if spins_index == 0:
                        self.ax.plot(
                                self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    else:
                        self.ax.plot(
                                self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )

        elif orientation == 'vertical':
            self.set_xlabel('DOS')
            self.set_ylabel('Energy (eV)')

            if len(spins) == 2:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            else:
                self.set_xlim([0,self.dos.total.max()])

            for spins_index , ispin in enumerate(spins):
                x = []
                y_total = []
                bar_color = []
                for idos in range(len(self.dos.energies)):
                    x.append(self.dos.energies[idos])
                    y = dos_projected[ispin][idos]
                    y_total.append(dos_total[ispin][idos])
                    y_total_projected = dos_total_projected[ispin][idos]
                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        y_total[-1] *= -1
                        y_total_projected *= -1

                    bar_color.append(cmap(y / (y_total_projected )))


                for idos in range(len(x) - 1):
                    self.ax.fill_betweenx([x[idos], x[idos + 1]],
                                    [y_total[idos], y_total[idos + 1]],
                                    color=bar_color[idos])

                if plot_total == True:
                    if spins_index == 0:
                        self.ax.plot(
                                self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    else:
                        self.ax.plot(
                                -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    
    def plot_parametric_line(self,
                             atoms:List[int]=None,
                             spins:List[int]=None,
                             principal_q_numbers:List[int]=[-1],
                             orbitals:List[int]=None,
                             spin_colors:List[str] or List[List[float]] =None,
                             spin_labels:List[str]=None,
                             vmin:float=None,
                             vmax:float=None,
                             plot_bar:bool=True,
                             cmap:str='jet',
                             orientation:str="horizontal",
                             ):
        """A method to plot the parametric line plot

        Parameters
        ----------
        atoms : List[int], optional
            A list of atoms, by default None
        spins : List[int], optional
            A list of spins, by default None
        principal_q_numbers : List[int], optional
            A list of principal quantum numbers, by default [-1]
        orbitals : List[int], optional
            A list of orbitals, by default None
        spin_colors : List[str] or List[List[float]], optional
            A list of spins colors, by default None
        spin_labels : List[str], optional
            A list of spin labels, by default None
        vmin : float, optional
            Value to normalize the minimum projection value., by default None
        vmax : float, optional
            Value to normalize the mmaximum projection value., by default None
        plot_bar : bool, optional
            Boolean to plot the colorbar, by default True
        cmap : str, optional
            The colormap to use, by default 'jet'
        orientation : str, optional
            String to plot vertical or horizontal, by default "horizontal"
        """

        if spin_colors is None:
            spin_colors = settings.dos.spin_colors
        if spin_labels is None:
            spin_labels = settings.dos.spin_labels

        if spins is None:
            if self.dos.is_non_collinear:
                spins = [0,1,2]
            else:
                spins = range(self.dos.n_spins)
        spin_projections = spins
        # This covers the non-colinear case when spins only represent projections.  
        if self.dos.is_non_collinear:
            spins = [0]

        dos_total_projected = self.dos.dos_sum()
        dos_projected = self.dos.dos_sum(atoms=atoms,
                               principal_q_numbers=principal_q_numbers,
                               orbitals=orbitals,
                               spins=spin_projections)

        projections_weights = np.divide(dos_projected,dos_total_projected)

        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = (dos_projected.max() / dos_total_projected.max())
        if plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax)

        if orientation == 'horizontal':

            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            if len(spins) == 2:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
            else:
                self.set_ylim([0,self.dos.total.max()])

            for spins_index , ispin in enumerate(spins):
                if len(spins)>1 and spins_index:
                    points = np.array( [self.dos.energies, -1 * self.dos.total[ispin, :]]).T.reshape(-1, 1, 2)
                else:
                    points = np.array( [self.dos.energies, self.dos.total[ispin, :]]).T.reshape(-1, 1, 2)

                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=plt.get_cmap(settings.ebs.color_map), norm=norm)
                lc.set_array(projections_weights[ispin,:])
                handle = self.ax.add_collection(lc)
                
                lc.set_linewidth(settings.dos.linewidth[ispin])
                lc.set_linestyle(settings.dos.linestyle[ispin])

                self.handles.append(handle)

        elif orientation == 'vertical':
            self.set_xlabel('DOS')
            self.set_ylabel('Energy (eV)')

            if len(spins) == 2:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            else:
                self.set_xlim([0,self.dos.total.max()])

            for spins_index , ispin in enumerate(spins):
                if len(spins)>1 and spins_index:
                    points = np.array( [ -1 * self.dos.total[ispin, :], self.dos.energies]).T.reshape(-1, 1, 2)
                else:
                    points = np.array( [self.dos.total[ispin, :], self.dos.energies]).T.reshape(-1, 1, 2)

                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=plt.get_cmap(settings.ebs.color_map), norm=norm)
                lc.set_array(projections_weights[ispin,:])
                handle = self.ax.add_collection(lc)
                
                lc.set_linewidth(settings.dos.linewidth[ispin])
                lc.set_linestyle(settings.dos.linestyle[ispin])
                self.handles.append(handle)

    def plot_stack_species(
            self,
            principal_q_numbers:List[int]=[-1],
            orbitals:List[int]=None,
            spins:List[int]=None,
            spin_colors:List[str] or List[List[float]] =None,
            spin_labels:List[str] = None,
            colors:List[str] or List[List[float]] =None,
            plot_total:bool=False,
            orientation:str="horizontal",
        ):
        """A method to plot the dos with the species contribution stacked on eachother

        Parameters
        ----------
        principal_q_numbers : List[int], optional
            A list of principal quantum numbers, by default [-1]
        orbitals : List[int], optional
            A list of orbitals, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_colors : List[str] or List[List[float]], optional
            A list of spin colors, by default None
        spin_labels : List[str], optional
            A list of spin labels, by default None
        colors : List[str] or List[List[float]], optional
            A list of colors, by default None
        plot_total : bool, optional
            Boolean to plot the total dos, by default False
        orientation : str, optional
            String to plot horizontal or vertical plot, by default "horizontal"
        """
        
        if spin_colors is None:
            spin_colors = settings.dos.spin_colors
        if spin_labels is None:
            spin_labels = settings.dos.spin_labels

        if spins is None:
            spins = range(self.dos.n_spins)
            if self.dos.is_non_collinear:
                spins = [0,1,2]
            else:
                spins = range(self.dos.n_spins)
        spin_projections = spins

        if self.dos.is_non_collinear:
            spins = [0]

        # This condition will depend on which orbital basis is being used.
        if self.dos.is_non_collinear and len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
            spins = [0]
            if orbitals:
                print("The plot only considers orbitals", orbitals)
                label = "-"
                if sum([x in orbitals for x in [0,1]]) == 2:
                    label += "s-j=0.5"
                if sum([x in orbitals for x in [2,3]]) == 2:
                    label += "p-j=0.5"
                if sum([x in orbitals for x in [4,5,6,7]]) == 4:
                    label += "p-j=1.5"
                if sum([x in orbitals for x in [8,9,10,11]]) == 4:
                    label += "d-j=1.5"
                if sum([x in orbitals for x in [12,13,14,15,16,17]]) == 6:
                    label += "d-j=2.5"
            else:
                if len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
                    label = "-spd-j=0.5,1.5,2.5"
                else:
                    label = "-"
        else:
            if orbitals:
                print("The plot only considers orbitals", orbitals)
                label = "-"
                if sum([x in orbitals for x in [0]]) == 1:
                    label += "s"
                if sum([x in orbitals for x in [1, 2, 3]]) == 3:
                    label += "p"
                if sum([x in orbitals for x in [4, 5, 6, 7, 8]]) == 5:
                    label += "d"
                if sum([x in orbitals for x in [9, 10, 11, 12, 13, 14, 15]]) == 7:
                    label += "f"
            else:
                if len(self.dos.projected[0][0]) == 1 + 3 + 5:
                    label = "-spd"
                elif len(self.dos.projected[0][0]) == 1 + 3 + 5 + 7:
                    label = "-spdf"
                else:
                    label = "-"

        if colors is None:
            colors = settings.dos.colors

        dos_total = self.dos.total
        dos_projected_total = self.dos.dos_sum()


        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS Cumlative')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            if len(spins) == 1:
                self.set_ylim([0,self.dos.total.max()])
            else:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])

            for spins_index , ispin in enumerate(spins):
                # bottom = np.zeros_like(self.dos.energies[cond])
                bottom = np.zeros_like(self.dos.energies)
                for specie in range(len(self.structure.species)):
                    idx = (np.array(self.structure.atoms) == self.structure.species[specie])
                    atoms = list(np.where(idx)[0])

                    dos_projected = self.dos.dos_sum(atoms=atoms,
                                        principal_q_numbers=principal_q_numbers,
                                        orbitals=orbitals,
                                        spins=spin_projections)

                    x = self.dos.energies
                    y = (dos_projected[ispin]  / dos_projected_total[ispin] ) * dos_total[ispin]

                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        handle = self.ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[specie],
                                        )
                    else:
                        handle = self.ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[specie],
                        )
                    self.handles.append(handle)
                    self.labels.append(self.structure.species[specie] + label + spin_labels[ispin])
                    bottom += y

                if plot_total == True:
                    if spins_index == 0:
                        self.ax.plot(
                                self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    else:
                        self.ax.plot(
                                self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )



        elif orientation == 'vertical':
            self.set_xlabel('DOS Cumlative')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            if len(spins) == 1:
                self.set_xlim([0,self.dos.total.max()])
            else:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

            for spins_index , ispin in enumerate(spins):
                # bottom = np.zeros_like(self.dos.energies[cond])
                bottom = np.zeros_like(self.dos.energies)
                for specie in range(len(self.structure.species)):
                    idx = (np.array(self.structure.atoms) == self.structure.species[specie])
                    atoms = list(np.where(idx)[0])

                    dos = self.dos.dos_sum(atoms=atoms,
                                        principal_q_numbers=principal_q_numbers,
                                        orbitals=orbitals,
                                        spins=spin_projections)

                    x = self.dos.energies
                    y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]

                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        handle = self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[specie],
                                        )
                    else:
                         handle = self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[specie],
                                        )
                    self.handles.append(handle)
                    self.labels.append(self.structure.species[specie] + label + spin_labels[ispin])     
                    bottom += y 

                if plot_total == True:
                    if spins_index == 0:
                        self.ax.plot(
                                self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    else:
                        self.ax.plot(
                                -self.dos.total[ispin, :], self.dos.energies,color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )


    def plot_stack_orbitals(self,
            atoms:List[int]=None,
            spins:List[int]=None,
            principal_q_numbers:List[int]=[-1],
            spin_colors:List[str] or List[List[float]] =None,
            spin_labels:List[str] = None,
            colors:List[str] or List[List[float]] =None,
            plot_total:bool= True,
            orientation:str="horizontal",
        ):
        """A method to plot dos orbitals contribution stacked.

        Parameters
        ----------
        atoms : List[int], optional
            A list of atoms, by default None
        spins : List[int], optional
            A list of spins, by default None
        principal_q_numbers : List[int], optional
            A list of principal quantum numbers, by default [-1]
        spin_colors : List[str] or List[List[float]], optional
            A list of spin colors, by default None
        spin_labels : List[str], optional
            A list of spin labels, by default None
        colors : List[str] or List[List[float]], optional
            A list of colors, by default None
        plot_total : bool, optional
            Boolean to plot the total dos, by default True
        orientation : str, optional
            String to plot horizontal or vertical, by default "horizontal"
        """


        if spin_colors is None:
            spin_colors = settings.dos.spin_colors
        if spin_labels is None:
            spin_labels = settings.dos.spin_labels

        if spins is None:
            spins = range(self.dos.n_spins)
            if self.dos.is_non_collinear:
                spins = [0,1,2]
            else:
                spins = range(self.dos.n_spins)
        spin_projections = spins
        if self.dos.is_non_collinear:
            spins = [0]

        atom_names = ""
        if atoms:
            print(
                "The plot only considers atoms",
                np.array(self.structure.atoms)[atoms],
            )
            atom_names = ""
            for ispc in np.unique(np.array(self.structure.atoms)[atoms]):
                atom_names += ispc + "-"
        all_atoms = ""
        for ispc in np.unique(np.array(self.structure.atoms)):
            all_atoms += ispc + "-"
        if atom_names == all_atoms:
            atom_names = ""

        if self.dos.is_non_collinear and len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
            orb_names = ["s-j=0.5", "p-j=0.5", "p-j=1.5", "d-j=1.5", "d-j=2.5"]
            orb_l = [[0,1], [2,3], [4, 5, 6, 7], [8,9,10,11], [12,13,14,15,16,17]]
        else:
            orb_names = ["s", "p", "d"]
            orb_l = [[0], [1, 2, 3], [4, 5, 6, 7, 8]]

        if colors is None:
            colors = settings.dos.colors
        
        dos_total = self.dos.total
        dos_projected_total = self.dos.dos_sum()

        if  orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS Cumlative')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            if len(spins) == 1:
                self.set_ylim([0,self.dos.total.max()])
            else:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])

            for spins_index , ispin in enumerate(spins):
                bottom = np.zeros_like(self.dos.energies)

                for iorb in range(len(orb_l)):
                    dos = self.dos.dos_sum(atoms=atoms,
                                        principal_q_numbers=principal_q_numbers,
                                        orbitals=orb_l[iorb],
                                        spins=spins)

                    x = self.dos.energies
                    y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]
                    y = np.nan_to_num(y, 0)

                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        handle =  self.ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[iorb])
                        
                    else:
                        handle = self.ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[iorb],
                        )
                        

                    self.labels.append(atom_names + orb_names[iorb] + spin_labels[ispin])
                    self.handles.append(handle)
                    bottom += y
            if plot_total == True:
                if ispin == 0:
                    self.ax.plot(
                            self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                            linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                        )
                else:
                    self.ax.plot(
                            self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                            linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                        )

        elif orientation == 'vertical':
            self.set_xlabel('DOS Cumlative')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            if len(spins) == 1:
                self.set_xlim([0,self.dos.total.max()])
            else:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

            for spins_index , ispin in enumerate(spins):
                bottom = np.zeros_like(self.dos.energies)

                for iorb in range(len(orb_l)):
                    dos = self.dos.dos_sum(atoms=atoms,
                                        principal_q_numbers=principal_q_numbers,
                                        orbitals=orb_l[iorb],
                                        spins=spins)

                    x = self.dos.energies
                    y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]
                    y = np.nan_to_num(y, 0)

                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        handle =  self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[iorb])
                        
                    else:
                        handle = self.ax.fill_betweenx(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[iorb],
                        )
                        

                    self.labels.append(atom_names + orb_names[iorb] + spin_labels[ispin])
                    self.handles.append(handle)
                    bottom += y
            if plot_total == True:
                if ispin == 0:
                    self.ax.plot(
                            self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                            linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                        )
                else:
                    self.ax.plot(
                            -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                            linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                        )
            
    def plot_stack(self,
                items:dict=None,
                spins:List[int]=None,
                spin_colors:List[str] or List[List[float]] =None,
                spin_labels:List[str]=None,
                colors:List[str] or List[List[float]] =None,
                plot_total:bool= True,
                orientation:str='horizontal',
        ):
        """A method to plot dos contributions stacked.

        Parameters
        ----------
        items : dict, optional
            A dictionary where the keys represent the atom and the 
            values are the orbital contributions to include for that item, by default None
        spins : List[int], optional
            A list of spins, by default None
        spin_colors : List[str] or List[List[float]], optional
            A list of spin colors, by default None
        spin_labels : List[str], optional
            A list of spin labels, by default None
        colors : List[str] or List[List[float]], optional
            A list of colors, by default None
        plot_total : bool, optional
            Boolean to plot the total dos, by default True
        orientation : str, optional
            String to plot horizontal or vertical, by default "horizontal"
        """
            
        if len(items) is None:
            print("""Please provide the stacking items in which you want
                to plot, example : {'Sr':[1,2,3],'O':[4,5,6,7,8]}
                will plot the stacked plots of p orbitals of Sr and
                d orbitals of Oxygen.""")
        if spin_colors is None:
            spin_colors = settings.dos.spin_colors
        if spin_labels is None:
            spin_labels = settings.dos.spin_labels
        
        if spins is None:
            spins = range(self.dos.n_spins)
            if self.dos.is_non_collinear:
                spins = [0,1,2]
            else:
                spins = range(self.dos.n_spins)
        spin_projections = spins
        if self.dos.is_non_collinear:
            spins = [0]
        
        if self.dos.is_non_collinear and len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
            if len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
                all_orbitals = "-spd-j=0.5,1.5,2.5"
            else:
                all_orbitals = "-"
        else:
            if len(self.dos.projected[0][0]) == (1 + 3 + 5):
                all_orbitals = "spd"
            elif len(self.dos.projected[0][0]) == (1 + 3 + 5 + 7):
                all_orbitals = "spdf"
            else:
                all_orbitals = ""

        if colors is None:
            colors = settings.dos.colors
        counter = 0
        colors_dict = {}
        for specie in items:
            colors_dict[specie] = colors[counter]
            counter += 1
        
        
        dos_total = self.dos.total
        dos_projected_total = self.dos.dos_sum()

        if orientation=='horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS Cumlative')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            if self.dos.n_spins == 2:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
            else:
                self.set_ylim([0,self.dos.total.max()])

            for ispin in spins:

                bottom = np.zeros_like(self.dos.energies)
                for specie in items:
                    idx = np.array(self.structure.atoms) == specie
                    atoms = list(np.where(idx)[0])
                    orbitals = items[specie]

                    dos = self.dos.dos_sum(atoms=atoms,
                                        spins=spin_projections,
                                        orbitals=orbitals)

                    label = "-"
                    # For coupled basis
                    if  len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
                        if sum([x in orbitals for x in [0,1]]) == 2:
                            label += "s-j=0.5"
                        if sum([x in orbitals for x in [2,3]]) == 2:
                            label += "p-j=0.5"
                        if sum([x in orbitals for x in [4,5,6,7]]) == 4:
                            label += "p-j=1.5"
                        if sum([x in orbitals for x in [8,9,10,11]]) == 4:
                            label += "d-j=1.5"
                        if sum([x in orbitals for x in [12,13,14,15,16,17]]) == 6:
                            label += "d-j=2.5"
                        if label == "-" + all_orbitals:
                            label = ""
                    # For uncoupled basis
                    else:
                        if sum([x in orbitals for x in [0]]) == 1:
                            label += "s"
                        if sum([x in orbitals for x in [1, 2, 3]]) == 3:
                            label += "p"
                        if sum([x in orbitals for x in [4, 5, 6, 7, 8]]) == 5:
                            label += "d"
                        if sum([x in orbitals
                                for x in [9, 10, 11, 12, 13, 14, 15]]) == 7:
                            label += "f"
                        if label == "-" + all_orbitals:
                            label = ""
                
                    x = self.dos.energies
                    y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]

                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        handle = self.ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors_dict[specie])
                    else:
                        handle = self.ax.fill_between(
                                x,
                                bottom + y,
                                bottom,
                                color=colors_dict[specie],
                            )
                    self.handles.append(handle)
                    self.labels.append(specie + label + spin_labels[ispin])
                    bottom += y
                if plot_total == True:
                    if ispin == 0:
                        self.ax.plot(
                                self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    else:
                        self.ax.plot(
                                    self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.dos.opacity[ispin], 
                                    linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                                )

        elif orientation=='vertical':
            self.set_xlabel('DOS Cumlative')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            if self.dos.n_spins == 2:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            else:
                self.set_xlim([0,self.dos.total.max()])
                
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

            for ispin in spins:

                bottom = np.zeros_like(self.dos.energies)
                for specie in items:
                    idx = np.array(self.structure.atoms) == specie
                    atoms = list(np.where(idx)[0])
                    orbitals = items[specie]

                    dos = self.dos.dos_sum(atoms=atoms,
                                        spins=spins,
                                        orbitals=orbitals)

                    label = "-"
                    # coupled basis
                    if  len(self.dos.projected[0][0]) == 2 + 2 + 4 + 4 + 6:
                        if sum([x in orbitals for x in [0,1]]) == 2:
                            label += "s-j=0.5"
                        if sum([x in orbitals for x in [2,3]]) == 2:
                            label += "p-j=0.5"
                        if sum([x in orbitals for x in [4,5,6,7]]) == 4:
                            label += "p-j=1.5"
                        if sum([x in orbitals for x in [8,9,10,11]]) == 4:
                            label += "d-j=1.5"
                        if sum([x in orbitals for x in [12,13,14,15,16,17]]) == 6:
                            label += "d-j=2.5"
                        if label == "-" + all_orbitals:
                            label = ""
                    # For uncoupled basis
                    else:
                        if sum([x in orbitals for x in [0]]) == 1:
                            label += "s"
                        if sum([x in orbitals for x in [1, 2, 3]]) == 3:
                            label += "p"
                        if sum([x in orbitals for x in [4, 5, 6, 7, 8]]) == 5:
                            label += "d"
                        if sum([x in orbitals
                                for x in [9, 10, 11, 12, 13, 14, 15]]) == 7:
                            label += "f"
                        if label == "-" + all_orbitals:
                            label = ""
                
                    x = self.dos.energies
                    y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]

                    if ispin > 0 and len(spins) > 1:
                        y *= -1
                        handle = self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors_dict[specie])
                    else:
                        handle = self.ax.fill_betweenx(
                                x,
                                bottom + y,
                                bottom,
                                color=colors_dict[specie],
                            )
                    self.handles.append(handle)
                    self.labels.append(specie + label + spin_labels[ispin])
                    bottom += y

                if plot_total == True:
                    if ispin == 0:
                        self.ax.plot(
                                self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
                    else:
                        self.ax.plot(
                                -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.dos.opacity[ispin], 
                                linestyle=settings.dos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.dos.linewidth[ispin],
                            )
        return None



    def set_xticks(self, 
                tick_positions:List[int]=None, 
                tick_names:List[str]=None):
        """A method to set the xticks of the plot

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None

        """

        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)
        return None
    def set_yticks(self,  
                    tick_positions:List[int]=None, 
                    tick_names:List[str]=None):
        """A method to set the yticks of the plot

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None

        """
        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)
        return None
        
    def set_xlim(self, 
                interval:List[int]=None):
        """A method to set the xlim of the plot

        Parameters
        ----------
        interval : List[int], optional
            The x interval, by default None
        """
        if interval is not None:
            self.ax.set_xlim(interval)
        return None

    def set_ylim(self, 
                interval:List[int]=None):
        """A method to set the ylim of the plot

        Parameters
        ----------
        interval : List[int], optional
            The y interval, by default None
        """
        if interval is not None:
            self.ax.set_ylim(interval)
        return None

    def set_xlabel(self, label:str):
        """A method to set the x label

        Parameters
        ----------
        label : str
            The x label name

        Returns
        -------
        None
            None
        """
        self.ax.set_xlabel(label)
        return None

    def set_ylabel(self, label:str):
        """A method to set the y label

        Parameters
        ----------
        label : str
            The y label name

        Returns
        -------
        None
            None
        """
        self.ax.set_ylabel(label)
    
    def legend(self, 
                labels:List[str]=None):
        """A method to include the legend

        Parameters
        ----------
        label : str
            The labels for the legend

        Returns
        -------
        None
            None
        """
        if labels == None:
            labels = self.labels
        self.ax.legend(self.handles, labels)
        return None

    def draw_fermi(self, 
                orientation:str='horizontal',
                color:str="blue", 
                linestyle:str="dotted", 
                linewidth:float=1):
        """A method to draw the fermi surface

        Parameters
        ----------
        orientation : str, optional
            Boolean to plot vertical or horizontal, by default 'horizontal'
        color : str, optional
            A color , by default "blue"
        linestyle : str, optional
            THe line style, by default "dotted"
        linewidth : float, optional
            The linewidth, by default 1

        Returns
        -------
        None
            None
        """
        if orientation == 'horizontal':
            self.ax.axvline(x=0, color=color, linestyle=linestyle, linewidth=linewidth)
        elif orientation == 'vertical':
            self.ax.axhline(y=0, color=color, linestyle=linestyle, linewidth=linewidth)
        return None
    def grid(self):
        """A method to include a grid on the plot.

        Returns
        -------
        None
            None
        """
        self.ax.grid(
            settings.dos.grid,
            which=settings.dos.grid_which,
            color=settings.dos.grid_color,
            linestyle=settings.dos.grid_linestyle,
            linewidth=settings.dos.grid_linewidth)
        return None

    def show(self):
        """A method to show the plot

        Returns
        -------
        None
            None
        """
        plt.show()
        return None

    def save(self, filename:str='dos.pdf'
        ):
        """A method to save the plot

        Parameters
        ----------
        filename : str, optional
            The filename, by default 'dos.pdf'

        Returns
        -------
        None
            None
        """

        plt.savefig(filename, bbox_inches="tight")
        plt.clf()
        return None
    
        

