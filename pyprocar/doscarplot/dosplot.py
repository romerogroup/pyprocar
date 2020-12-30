"""
Created on May 17 2020
@author: Pedram Tavadze
"""

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
figsize = (12, 6)


class DosPlot:
    def __init__(self, dos=None, structure=None):
        """

        Parameters
        ----------
        vaspxml : TYPE, optional
            DESCRIPTION. The default is "vasprun.xml".
        Returns
        -------
        None.
        """
        self.dos = dos
        self.structure = structure

        return

    def plot_total(self,
                   spins=None,
                   spin_colors=None,
                   figsize=figsize,
                   ax=None,
                   orientation="horizontal",
                   labels=None,
                   linewidth=1,
                   ):
        """

        Parameters
        ----------
        spins : TYPE, optional
            DESCRIPTION. The default is None.
        spin_colors : TYPE, optional
            DESCRIPTION. The default is None.
        figsize : TYPE, optional
            DESCRIPTION. The default is figsize.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        orientation : TYPE, optional
            DESCRIPTION. The default is "horizontal".
        labels : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.
        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        """

        if spin_colors is None:
            spin_colors = [(1, 0, 0), (0, 0, 1)]

        energies = self.dos.energies
        dos = np.array(self.dos.total)

        if spins is None:
            spins = np.arange(len(self.dos.total))

        fig, ax = plotter(energies, dos, spins, spin_colors, figsize, ax,
                          orientation, linewidth, labels)
        return fig, ax

    def plot_parametric_line(self,
                             atoms=None,
                             principal_q_numbers=[-1],
                             orbitals=None,
                             spins=None,
                             spin_colors=None,
                             figsize=(12, 6),
                             ax=None,
                             orientation="horizontal",
                             labels=None,
                             linewidth=1,
                             ):

        if ax is None:
            if orientation == "horizontal":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            elif orientation == "vertical":
                fig = plt.figure(figsize=(figsize[1], figsize[0]))
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        if spin_colors is None:
            spin_colors = [(1, 0, 0), (0, 0, 1)]

        # dos = self.parsedData.dos_parametric(atoms=atoms,
        # spin=spins,
        # orbitals=orbitals)

        if spins is None:
            spins = np.arange(len(self.dos.total))

        dos = self.dos.dos_sum(atoms, principal_q_numbers, orbitals, spins)

        fig, ax = plotter(self.dos.energies, dos, spins, spin_colors, figsize,
                          ax, orientation, linewidth, labels)
        return fig, ax

    def plot_parametric(self,
                        atoms=None,
                        principal_q_numbers=[-1],
                        orbitals=None,
                        spins=None,
                        spin_colors=None,
                        cmap="jet",
                        vmin=0,
                        vmax=1,
                        elimit=None,
                        figsize=(12, 6),
                        ax=None,
                        orientation="horizontal",
                        labels=None,
                        plot_bar=True):

        if ax is None:
            if orientation == "horizontal":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            elif orientation == "vertical":
                fig = plt.figure(figsize=(figsize[1], figsize[0]))
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        if spin_colors is None:
            spin_colors = [(1, 0, 0), (0, 0, 1)]
        cmap = mpl.cm.get_cmap(cmap)

        dos_total = np.array(self.dos.total)
        dos_total_projected = self.dos.dos_sum()
        if spins is None:
            spins = np.arange(len(self.dos.total))

        dos = self.dos.dos_sum(atoms=atoms,
                               principal_q_numbers=principal_q_numbers,
                               orbitals=orbitals,
                               spins=spins)
        # dos_total = dos
        # dos_total = self.parsedData.dos_total
        # dos_total_projected = self.parsedData.dos_parametric()
        # dos = self.parsedData.dos_parametric(atoms=atoms,
        #                                     spin=spins,
        #                                      orbitals=orbitals)
        if vmin is None or vmax is None:
            vmin = 0
            vmax = (dos.max() / dos_total.max())
        if plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        if not elimit:
            elimit = [self.dos.energies.min(), self.dos.energies.max()]
        deltaE = elimit[1]-elimit[0]

        cond1 = self.dos.energies >= elimit[0] - deltaE*0.05
        cond2 = self.dos.energies <= elimit[1] + deltaE*0.05
        cond = np.all([cond1, cond2], axis=0)

        # dE = dos.energies[1] - dos.energies[0]
        if spins is None:
            spins = np.arange(len(self.dos.total))
        for ispin in spins:
            x = []
            y_total = []
            bar_color = []
            for idos in range(len(self.dos.energies[cond])):
                x.append(self.dos.energies[cond][idos])
                y = dos[ispin, cond][idos]

                y_total.append(dos_total[ispin, cond][idos])
                y_total_projected = dos_total_projected[ispin, cond][idos]
                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    y_total[-1] *= -1
                    y_total_projected *= -1

                bar_color.append(cmap(y / (y_total_projected * (vmax - vmin))))
            if orientation == "horizontal":
                #                ax.bar(x,y_total,dE,color=bar_color)
                for idos in range(len(x) - 1):
                    ax.fill_between([x[idos], x[idos + 1]],
                                    [y_total[idos], y_total[idos + 1]],
                                    color=bar_color[idos])
            elif orientation == "vertical":
                for idos in range(len(x) - 1):
                    ax.fill_betweenx([x[idos], x[idos + 1]],
                                     [y_total[idos], y_total[idos + 1]],
                                     color=bar_color[idos])

        return fig, ax

    def plot_stack_species(
            self,
            principal_q_numbers=[-1],
            orbitals=None,
            spins=None,
            spin_colors=None,
            colors=None,
            elimit=None,
            figsize=(12, 6),
            ax=None,
            orientation="horizontal",
    ):
        if spin_colors is None:
            spin_colors = [(0, 0, 1), (1, 0, 0)]
        if colors is None:
            colors = [
                'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange',
                'purple', 'brown', 'navy', 'maroon', 'olive'
            ]

        if ax is None:
            if orientation == "horizontal":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            elif orientation == "vertical":
                fig = plt.figure(figsize=(figsize[1], figsize[0]))
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        if spins is None:
            spins = np.arange(len(self.dos.total))

        if not elimit:
            elimit = [self.dos.energies.min(), self.dos.energies.max()]
        # dos_projected_total = self.parsedData.dos_parametric(spin=spins,
        # orbitals=orbitals)

        if len(self.dos.projected[0][0]) == 1 + 3 + 5:
            all_orbitals = "spd"
        elif len(self.dos.projected[0][0]) == 1 + 3 + 5 + 7:
            all_orbitals = "spdf"
        else:
            all_orbitals = ""

        label = ""
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
            if label == "-" + all_orbitals:
                label = ""
        # dos_total = self.parsedData.dos_total

        # cond1 = self.parsedData.dos_total.energies >= elimit[0]
        # cond2 = self.parsedData.dos_total.energies <= elimit[1]
        # cond = np.all([cond1, cond2], axis=0)

        dos_total = self.dos.total
        dos_projected_total = self.dos.dos_sum()

        deltaE = elimit[1]-elimit[0]

        cond1 = self.dos.energies >= elimit[0] - deltaE*0.05
        cond2 = self.dos.energies <= elimit[1] + deltaE*0.05
        cond = np.all([cond1, cond2], axis=0)

        for ispin in spins:
            # bottom = np.zeros_like(self.parsedData.dos_total.energies[cond])
            bottom = np.zeros_like(self.dos.energies[cond])
            for ispc in range(len(self.structure.species)):
                idx = (np.array(
                    self.structure.atoms) == self.structure.species[ispc])
                atoms = list(np.where(idx)[0])
                # dos = self.parsedData.dos_parametric(atoms=atoms,
                #                                      spin=spins,
                #                                      orbitals=orbitals)

                dos = self.dos.dos_sum(atoms=atoms,
                                       principal_q_numbers=principal_q_numbers,
                                       orbitals=orbitals,
                                       spins=spins)

                x = self.dos.energies[cond]
                y = (dos[ispin, cond] *
                     dos_total[ispin, cond]) / dos_projected_total[ispin, cond]

                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    if orientation == "horizontal":
                        ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[ispc])
                    elif orientation == "vertical":
                        ax.fill_betweenx(x,
                                         bottom + y,
                                         bottom,
                                         color=colors[ispc])
                else:
                    if orientation == "horizontal":
                        ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[ispc],
                            label=self.structure.species[ispc] + label,
                        )
                    elif orientation == "vertical":
                        ax.fill_betweenx(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[ispc],
                            label=self.structure.species[ispc] + label,
                        )
                bottom += y
        return fig, ax

    def plot_stack_orbitals(
            self,
            atoms=None,
            principal_q_numbers=[-1],
            spins=None,
            spin_colors=None,
            colors=None,
            elimit=None,
            figsize=(12, 6),
            ax=None,
            orientation="horizontal",
    ):
        if spin_colors is None:
            spin_colors = [(0, 0, 0), (0, 0, 0)]
        if colors is None:
            colors = [
                'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange',
                'purple', 'brown', 'navy', 'maroon', 'olive'
            ]

        if spins is None:
            spins = np.arange(len(self.dos.total), dtype=np.int)
        if ax is None:
            if orientation == "horizontal":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            elif orientation == "vertical":
                fig = plt.figure(figsize=(figsize[1], figsize[0]))
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
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

        if not elimit:
            elimit = [self.dos.energies.min(), self.dos.energies.max()]
        # dos_projected_total = self.parsedData.dos_parametric()
        deltaE = elimit[1]-elimit[0]

        cond1 = self.dos.energies >= elimit[0] - deltaE*0.05
        cond2 = self.dos.energies <= elimit[1] + deltaE*0.05
        cond = np.all([cond1, cond2], axis=0)

        orb_names = ["s", "p", "d"]
        orb_l = [[0], [1, 2, 3], [4, 5, 6, 7, 8]]
        for ispin in spins:
            bottom = np.zeros_like(self.dos.energies[cond])
            for iorb in range(3):
                # dos = self.parsedData.dos_parametric(atoms=atoms,
                #                                      spin=spins,
                #                                      orbitals=orb_l[iorb])
                dos = self.dos.dos_sum(atoms=atoms,
                                       principal_q_numbers=principal_q_numbers,
                                       orbitals=orb_l[iorb],
                                       spins=spins)
                x = self.dos.energies[cond]
                y = (dos[ispin, cond] *
                     dos_total[ispin, cond]) / dos_projected_total[ispin, cond]
                y = np.nan_to_num(y, 0)

                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    if orientation == "horizontal":
                        ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[iorb])
                    elif orientation == "vertical":
                        ax.fill_betweenx(x,
                                         bottom + y,
                                         bottom,
                                         color=colors[iorb])
                else:
                    if orientation == "horizontal":
                        ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[iorb],
                            label=atom_names + orb_names[iorb],
                        )
                    elif orientation == "vertical":
                        ax.fill_betweenx(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[iorb],
                            label=atom_names + orb_names[iorb],
                        )
                bottom += y
        return fig, ax

    def plot_stack(
            self,
            items={},
            spins=None,
            spin_colors=None,
            colors=None,
            elimit=None,
            figsize=(12, 6),
            ax=None,
            orientation="horizontal",
    ):

        if len(items) == 0:
            print("""Please provide the stacking items in which you want
                to plot, example : {'Sr':[1,2,3],'O':[4,5,6,7,8]}
                will plot the stacked plots of p orbitals of Sr and
                d orbitals of Oxygen.""")

        if spin_colors is None:
            spin_colors = [(0, 0, 1), (1, 0, 0)]
        src_colors = colors
        if src_colors is None:
            src_colors = [
                'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange',
                'purple', 'brown', 'navy', 'maroon', 'olive'
            ]
        if spins is None:
            spins = np.arange(len(self.dos.total), dtype=np.int)

        if ax is None:
            if orientation == "horizontal":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            elif orientation == "vertical":
                fig = plt.figure(figsize=(figsize[1], figsize[0]))
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        if not elimit:
            elimit = [self.dos.energies.min(), self.dos.energies.max()]
        # dos_total = self.parsedData.dos_total
        dos_total = self.dos.total

        # if self.dos.dos_projected[0].ncols == (1 + 3 +
        #                                               5) * dos_total.ncols:
        #     all_orbitals = "spd"
        # elif self.parsedData.dos_projected[0].ncols == (1 + 3 + 5 +
        #                                                 7) * dos_total.ncols:
        #     all_orbitals = "spdf"
        # else:
        #     all_orbitals = ""

        if len(self.dos.projected[0][0]) == (1 + 3 + 5):
            all_orbitals = "spd"
        elif len(self.dos.projected[0][0]) == (1 + 3 + 5 + 7):
            all_orbitals = "spdf"
        else:
            all_orbitals = ""
        deltaE = elimit[1]-elimit[0]

        cond1 = self.dos.energies >= elimit[0] - deltaE*0.05
        cond2 = self.dos.energies <= elimit[1] + deltaE*0.05
        cond = np.all([cond1, cond2], axis=0)
        counter = 0
        colors = {}

        for ispc in items:
            colors[ispc] = src_colors[counter]
            counter += 1

        # dos_projected_total = self.parsedData.dos_parametric(spin=spins)
        dos_projected_total = self.dos.dos_sum(spins=spins)

        for ispin in spins:
            bottom = np.zeros_like(self.dos.energies[cond])
            # bottom = np.zeros_like(self.parsedData.dos_total.energies[cond])
            # bottom = np.zeros_like(self.VaspXML.dos_total.energies[:])
            for ispc in items:
                idx = np.array(self.structure.atoms) == ispc
                atoms = list(np.where(idx)[0])
                orbitals = items[ispc]

                # dos = self.parsedData.dos_parametric(atoms=atoms,
                #                                      spin=spins,
                #                                      orbitals=orbitals)
                dos = self.dos.dos_sum(atoms=atoms,
                                       spins=spins,
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
                x = self.dos.energies[cond]
                # x = dos.energies[:]

                y = (dos[ispin, cond] *
                     dos_total[ispin, cond]) / dos_projected_total[ispin, cond]
                # y = ( dos.dos[:, ispin + 1] * dos_total.dos[:, ispin + 1]
                # ) / dos_projected_total.dos[:, ispin + 1]
                # y =  dos.dos[cond, ispin + 1]

                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    if orientation == "horizontal":
                        ax.fill_between(x,
                                        bottom + y,
                                        bottom,
                                        color=colors[ispc])
                    elif orientation == "vertical":
                        ax.fill_betweenx(x,
                                         bottom + y,
                                         bottom,
                                         color=colors[ispc])
                else:
                    if orientation == "horizontal":
                        ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[ispc],
                            label=ispc + label,
                        )
                    elif orientation == "vertical":

                        ax.fill_betweenx(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[ispc],
                            label=ispc + label,
                        )
                bottom += y
        return fig, ax


def plotter(energies,
            dos,
            spins,
            spin_colors,
            figsize,
            ax,
            orientation,
            linewidth,
            labels=None,
            ):

    if spins is None:
        spins = np.arange(dos.shape[0])

    if orientation == "horizontal":
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        x = energies

        for iy in spins:

            y = dos[iy, :]
            if iy > 0 and len(spins) > 1:
                y *= -1
            if labels is not None:
                ax.plot(x, y, "r-", color=spin_colors[iy], label=labels[iy], linewidth=linewidth)
            else:
                ax.plot(x, y, "r-", color=spin_colors[iy], linewidth=linewidth)

    elif orientation == "vertical":
        if ax is None:
            fig = plt.figure(figsize=(figsize[1], figsize[0]))
            ax = fig.add_subplot(111)
        else:
            fig = plt.gca()
        y = energies
        for ix in spins:
            x = dos[ix, :]
            if ix > 0 and len(spins) > 1:
                x *= -1
            if labels is not None:
                ax.plot(x, y, "r-", color=spin_colors[ix], label=labels[ix], linewidth=linewidth)
            else:
                ax.plot(x, y, "r-", color=spin_colors[ix], linewidth=linewidth)
    return fig, ax
