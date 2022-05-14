"""
Created on May 17 2020
@author: Pedram Tavadze
"""

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from ..utils.defaults import settings
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
np.seterr(divide="ignore", invalid="ignore")

# TODO add python typing to all of the functions
# TODO remove pyprocar.doscarplot module (redundant)


class DOSPlot:
    def __init__(self, dos=None, structure=None, spins=None, ax=None, **kwargs):
        """
        class to plot an electronic band structure.

        Parameters
        ----------
        dos : object
            An density of states pyprocar.core.DensityOfStates.
        structure : object
            An density of states pyprocar.core.Structure.
        
        ax : object, optional
            A matplotlib Axes object. If provided the plot will be located at that ax.
            The default is None.

        Returns
        -------
        None.

        """
        settings.modify(kwargs)

        self.dos = dos
        self.structure = structure
        
        if spins is None:
            self.spins = np.arange(self.dos.nspins, dtype=int)
            self.nspins = len(self.spins)
            
        else:
            self.spins = spins
            self.nspins = len(self.spins)

        if self.dos.is_non_collinear:
            self.spins = [0]

        self.handles = []
        self.labels = []
        if ax is None:
            self.fig = plt.figure(figsize=tuple(settings.general.figure_size),)
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax
    
        return


    def plot_dos(self, orientation = 'horizontal'):
        """
        Plot the plain density of states.

        Parameters
        ----------
        spins : list, optional
            A list of the spins to be plotted. The default is None.
        color : string, optional
            Color for the bands. The default is "blue".
        opacity : float, optional
            Opacity level between 0.0 and 1.0. The default is 1.0.

        Returns
        -------
        None.

        """
        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
        elif orientation == 'vertical':
            self.set_xlabel('DOS')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

        for ispin in range(len(self.spins)):
            # for iband in range(self.edos.nbands):
            if orientation == 'horizontal':
                handle = self.ax.plot(
                    self.dos.energies, self.dos.total[ispin, :], color=settings.edos.spin_colors[ispin], alpha=settings.edos.opacity[
                        ispin], linestyle=settings.edos.linestyle[ispin], label=settings.edos.spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                )
            elif orientation == 'vertical':
                handle = self.ax.plot(
                        self.dos.total[ispin, :], self.dos.energies, color=settings.edos.spin_colors[ispin], alpha=settings.edos.opacity[
                        ispin], linestyle=settings.edos.linestyle[ispin], label= settings.edos.spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                )
            self.handles.append(handle)

    def plot_parametric(self,
                        atoms=None,
                        principal_q_numbers=[-1],
                        orbitals=None,
                        spin_colors=None,
                        spin_labels=None,
                        cmap="jet",
                        vmin=0,
                        vmax=1,
                        plot_total=True,
                        plot_bar=True,
                        orientation = 'horizontal'):

        if spin_colors is None:
            spin_colors = settings.edos.spin_colors
        if spin_labels is None:
            spin_labels = settings.edos.spin_labels
        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            if len(self.spins) == 1:
                self.set_ylim([0,self.dos.total.max()])
            else:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
        elif orientation == 'vertical':
            self.set_xlabel('DOS')
            self.set_ylabel('Energy (eV)')
            if len(self.spins) == 1:
                self.set_xlim([0,self.dos.total.max()])
            else:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])
        
        cmap = mpl.cm.get_cmap(cmap)

        dos_total = np.array(self.dos.total)
        dos_total_projected = self.dos.dos_sum()
 
        dos = self.dos.dos_sum(atoms=atoms,
                               principal_q_numbers=principal_q_numbers,
                               orbitals=orbitals,
                               spins=self.spins)
     
        if vmin is None or vmax is None:
            vmin = 0
            vmax = (dos.max() / dos_total.max())
        if plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax)
        
        for ispin in self.spins:
            x = []
            y_total = []
            bar_color = []
            for idos in range(len(self.dos.energies)):
                x.append(self.dos.energies[idos])
                y = dos[ispin][idos]
                y_total.append(dos_total[ispin][idos])
                y_total_projected = dos_total_projected[ispin][idos]
                if ispin > 0 and len(self.spins) > 1:
                    y *= -1
                    y_total[-1] *= -1
                    y_total_projected *= -1

                bar_color.append(cmap(y / (y_total_projected * (vmax - vmin))))

            for idos in range(len(x) - 1):
                if orientation == 'horizontal':
                    self.ax.fill_between([x[idos], x[idos + 1]],
                                    [y_total[idos], y_total[idos + 1]],
                                    color=bar_color[idos])
                elif orientation == 'vertical':
                    self.ax.fill_betweenx([x[idos], x[idos + 1]],
                                     [y_total[idos], y_total[idos + 1]],
                                     color=bar_color[idos])
                    
            
            if plot_total == True:
                if ispin == 0:
                    if orientation == 'horizontal':
                        self.ax.plot(
                                self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )
                    elif orientation == 'vertical':
                        self.ax.plot(
                                self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )
                else:
                    if orientation == 'horizontal':
                        self.ax.plot(
                                self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )
                    elif orientation == 'vertical':
                        self.ax.plot(
                                -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )

    def plot_parametric_line(self,
                             atoms=None,
                             principal_q_numbers=[-1],
                             orbitals=None,
                             spin_colors=None,
                             spin_labels=None,
                             orientation="horizontal",
                             ):

        if spin_colors is None:
            spin_colors = settings.edos.spin_colors
        if spin_labels is None:
            spin_labels = settings.edos.spin_labels
        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            self.set_ylim([0,self.dos.total.max()])

        elif orientation == 'vertical':
            self.set_xlabel('DOS')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            self.set_xlim([0,self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

        dos_summed = self.dos.dos_sum(atoms, principal_q_numbers, orbitals, self.spins)
        
        for ispin in range(len(self.spins)):
            # for iband in range(self.edos.nbands):
            if orientation == 'horizontal':
                handle = self.ax.plot(
                    self.dos.energies, dos_summed[ispin, :], color=spin_colors[ispin], alpha=settings.edos.opacity[ispin], 
                    linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                )
                
            elif orientation == 'vertical':
                handle = self.ax.plot(
                    dos_summed[ispin, :], self.dos.energies, color=spin_colors[ispin], alpha=settings.edos.opacity[ispin], 
                    linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                )   
            self.handles.append(handle)

    def plot_stack_species(
            self,
            principal_q_numbers=[-1],
            orbitals=None,
            spin_colors=None,
            spin_labels = None,
            colors=None,
            plot_total=False,
            orientation="horizontal",
    ):

        if spin_colors is None:
            spin_colors = settings.edos.spin_colors
        if spin_labels is None:
            spin_labels = settings.edos.spin_labels
        if colors is None:
            colors = settings.edos.colors
        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS Cumlative')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            if len(self.spins) == 1:
                self.set_ylim([0,self.dos.total.max()])
            else:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
        elif orientation == 'vertical':
            self.set_xlabel('DOS Cumlative')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            if len(self.spins) == 1:
                self.set_xlim([0,self.dos.total.max()])
            else:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

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
            
        dos_total = self.dos.total
        dos_projected_total = self.dos.dos_sum()

        # elimit = [self.dos.energies.min(), self.dos.energies.max()]
        # deltaE = elimit[1]-elimit[0]
        # cond1 = self.dos.energies >= elimit[0] - deltaE*0.05
        # cond2 = self.dos.energies <= elimit[1] + deltaE*0.05
        # cond = np.all([cond1, cond2], axis=0)
        
        for ispin in self.spins:
            # bottom = np.zeros_like(self.dos.energies[cond])
            bottom = np.zeros_like(self.dos.energies)
            for specie in range(len(self.structure.species)):
                idx = (np.array(self.structure.atoms) == self.structure.species[specie])
                atoms = list(np.where(idx)[0])

                dos = self.dos.dos_sum(atoms=atoms,
                                       principal_q_numbers=principal_q_numbers,
                                       orbitals=orbitals,
                                       spins=self.spins)

                # x = self.dos.energies[cond]
                # y = (dos[ispin, cond] *
                #      dos_total[ispin, cond]) / dos_projected_total[ispin, cond]

                x = self.dos.energies
                y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]

                if ispin > 0 and len(self.spins) > 1:
                    y *= -1
                    if orientation == 'horizontal':
                        handle = self.ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[specie],
                                        )
                    elif orientation == 'vertical':
                        handle = self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[specie],
                                        )
                    
                else:
                    if orientation == 'horizontal':
                        handle = self.ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[specie],
                        )
                    elif orientation == 'vertical':
                        handle = self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[specie],
                                        )
                    

                bottom += y

                self.handles.append(handle)
                self.labels.append(self.structure.species[specie] + label + spin_labels[ispin])

            if plot_total == True:
                if ispin == 0:
                    if orientation == 'horizontal':
                        self.ax.plot(
                                self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )
                    elif orientation == 'vertical':
                        self.ax.plot(
                                self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )
                else:
                    if orientation == 'horizontal':
                        self.ax.plot(
                                self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )
                    elif orientation == 'vertical':
                        self.ax.plot(
                                -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                                linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                            )

    def plot_stack_orbitals(self,
            atoms=None,
            principal_q_numbers=[-1],
            spin_colors=None,
            spin_labels = None,
            colors=None,
            plot_total = True,
            orientation="horizontal",
    ):

        if spin_colors is None:
            spin_colors = settings.edos.spin_colors
        if spin_labels is None:
            spin_labels = settings.edos.spin_labels
        if colors is None:
            colors = settings.edos.colors
        
        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS Cumlative')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            if len(self.spins) == 1:
                self.set_ylim([0,self.dos.total.max()])
            else:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
        elif orientation == 'vertical':
            self.set_xlabel('DOS Cumlative')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            if len(self.spins) == 1:
                self.set_xlim([0,self.dos.total.max()])
            else:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])
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
        
        dos_total = self.dos.total
        dos_projected_total = self.dos.dos_sum()

        orb_names = ["s", "p", "d"]
        orb_l = [[0], [1, 2, 3], [4, 5, 6, 7, 8]]
        for ispin in self.spins:
            bottom = np.zeros_like(self.dos.energies)

            for iorb in range(3):
                dos = self.dos.dos_sum(atoms=atoms,
                                       principal_q_numbers=principal_q_numbers,
                                       orbitals=orb_l[iorb],
                                       spins=self.spins)

                x = self.dos.energies
                y = (dos[ispin] * dos_total[ispin]) / dos_projected_total[ispin]
                y = np.nan_to_num(y, 0)

                if ispin > 0 and len(self.spins) > 1:
                    y *= -1
                    if orientation == 'horizontal':
                        handle =  self.ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[iorb])
                    elif orientation == 'vertical':
                        handle =  self.ax.fill_betweenx(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[iorb])
                    
                else:
                    if orientation == 'horizontal':
                        handle = self.ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[iorb],
                        )
                    elif orientation == 'vertical':
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
                if orientation == 'horizontal':
                    self.ax.plot(
                            self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
                elif orientation == 'vertical':
                    self.ax.plot(
                            self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
            else:
                if orientation == 'horizontal':
                    self.ax.plot(
                            self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
                elif orientation == 'vertical':
                    self.ax.plot(
                            -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
            
    def plot_stack(self,
                    items=None,
                    spin_colors=None,
                    spin_labels=None,
                    colors = None,
                    plot_total = True,
                    orientation=None,
                ):
        if len(items) is None:
            print("""Please provide the stacking items in which you want
                to plot, example : {'Sr':[1,2,3],'O':[4,5,6,7,8]}
                will plot the stacked plots of p orbitals of Sr and
                d orbitals of Oxygen.""")
        if spin_colors is None:
            spin_colors = settings.edos.spin_colors
        if spin_labels is None:
            spin_labels = settings.edos.spin_labels
        if colors is None:
            colors = settings.edos.colors
        if orientation == 'horizontal':
            self.set_xlabel('Energy (eV)')
            self.set_ylabel('DOS Cumlative')
            self.set_xlim([self.dos.energies.min(),self.dos.energies.max()])
            self.set_ylim([self.dos.total.min(),self.dos.total.max()])
            if len(self.spins) == 1:
                self.set_ylim([0,self.dos.total.max()])
            else:
                self.set_ylim([-self.dos.total.max(),self.dos.total.max()])
        elif orientation == 'vertical':
            self.set_xlabel('DOS Cumlative')
            self.set_ylabel('Energy (eV)')
            self.set_xlim([self.dos.total.min(),self.dos.total.max()])
            if len(self.spins) == 1:
                self.set_xlim([0,self.dos.total.max()])
            else:
                self.set_xlim([-self.dos.total.max(),self.dos.total.max()])
            self.set_ylim([self.dos.energies.min(),self.dos.energies.max()])

        dos_total = self.dos.total
        if len(self.dos.projected[0][0]) == (1 + 3 + 5):
            all_orbitals = "spd"
        elif len(self.dos.projected[0][0]) == (1 + 3 + 5 + 7):
            all_orbitals = "spdf"
        else:
            all_orbitals = ""

        counter = 0
        colors_dict = {}
        for specie in items:
            colors_dict[specie] = colors[counter]
            counter += 1

        dos_projected_total = self.dos.dos_sum(spins=self.spins)
        for ispin in self.spins:

            bottom = np.zeros_like(self.dos.energies)
            for specie in items:
                idx = np.array(self.structure.atoms) == specie
                atoms = list(np.where(idx)[0])
                orbitals = items[specie]

                dos = self.dos.dos_sum(atoms=atoms,
                                       spins=self.spins,
                                       orbitals=orbitals)

                label = "-"
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

                if ispin > 0 and len(self.spins) > 1:
                    y *= -1
                    if orientation == 'horizontal':
                        handle = self.ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors_dict[specie])
                    elif orientation == 'vertical':
                        handle = self.ax.fill_betweenx(x,
                                    bottom + y,
                                    bottom,
                                    color=colors_dict[specie])
                else:
                    if orientation == 'horizontal':
                        handle = self.ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors_dict[specie],
                        )
                    elif orientation == 'vertical':
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
                if orientation == 'horizontal':
                    self.ax.plot(
                            self.dos.energies, self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
                elif orientation == 'vertical':
                    self.ax.plot(
                            self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
            else:
                if orientation == 'horizontal':
                    self.ax.plot(
                            self.dos.energies, -self.dos.total[ispin, :], color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )
                elif orientation == 'vertical':
                    self.ax.plot(
                            -self.dos.total[ispin, :], self.dos.energies, color= 'black', alpha=settings.edos.opacity[ispin], 
                            linestyle=settings.edos.linestyle[ispin], label=spin_labels[ispin], linewidth=settings.edos.linewidth[ispin],
                        )

    def set_xticks(self, tick_positions=None, tick_names=None, color="black"):
        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)
        
    def set_yticks(self,  tick_positions=None, tick_names=None, color="black"):
        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)
        
    def set_xlim(self, interval=None):
        if interval is not None:
            self.ax.set_xlim(interval)

    def set_ylim(self, interval=None):
        if interval is not None:
            self.ax.set_ylim(interval)

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)

    def set_ylabel(self, label):
        self.ax.set_ylabel(label)
    
    def legend(self, labels=None):
        if labels == None:
            labels = self.labels
        self.ax.legend(self.handles, labels)

    def draw_fermi(self, orientation = 'horizontal',color="blue", linestyle="dotted", linewidth=1):
        if orientation == 'horizontal':
            self.ax.axvline(x=0, color=color, linestyle=linestyle, linewidth=linewidth)
        elif orientation == 'vertical':
            self.ax.axhline(y=0, color=color, linestyle=linestyle, linewidth=linewidth)

    def grid(self):
        self.ax.grid(
            settings.edos.grid,
            which=settings.edos.grid_which,
            color=settings.edos.grid_color,
            linestyle=settings.edos.grid_linestyle,
            linewidth=settings.edos.grid_linewidth)
           
    def show(self):
        plt.show()

    def save(self, filename='dos.pdf'):        
        plt.savefig(filename, bbox_inches="tight")
        plt.clf()
    
        

