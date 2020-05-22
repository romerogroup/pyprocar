# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


class DosPlot:

    from pychemia.code.vasp import VaspXML

    def __init__(self, vaspxml="vasprun.xml"):
        self.VaspXML = VaspXML(vaspxml)
        return

    def plot_total(
        self,
        spins=None,
        markersize=0.02,
        marker="o",
        spin_colors=None,
        figsize=(12, 6),
        ax=None,
        orientation="horizontal",
        labels=None,
    ):

        # dos is a pychemia density of states object
        # pychemia.visual.DensityOfStates
        if spin_colors is None:
            spin_colors = [(1, 0, 0), (0, 0, 1)]
        dos = self.VaspXML.dos_total

        fig, ax = plotter(
            dos,
            spins,
            markersize,
            marker,
            spin_colors,
            figsize,
            ax,
            orientation,
            labels,
        )

        return fig, ax

    def plot_parametric_line(
        self,
        atoms=None,
        spins=None,
        orbitals=None,
        markersize=0.02,
        marker="o",
        spin_colors=None,
        figsize=(12, 6),
        ax=None,
        orientation="horizontal",
        labels=None,
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
        dos = self.VaspXML.dos_parametric(atoms=atoms, spin=spins, orbitals=orbitals)

        fig, ax = plotter(
            dos,
            spins,
            markersize,
            marker,
            spin_colors,
            figsize,
            ax,
            orientation,
            labels,
        )
        return fig, ax

    def plot_parametric(
        self,
        atoms=None,
        spins=None,
        orbitals=None,
        markersize=0.02,
        marker="o",
        spin_colors=None,
        cmap="jet",
        vmin=0,
        vmax=1,
        elimit=None,
        figsize=(12, 6),
        ax=None,
        orientation="horizontal",
        labels=None,
        plot_bar=True,
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
        cmap = mpl.cm.get_cmap(cmap)
        dos_total = self.VaspXML.dos_total
        dos_total_projected = self.VaspXML.dos_parametric()
        dos = self.VaspXML.dos_parametric(atoms=atoms, spin=spins, orbitals=orbitals)

        if ax is None:
            if orientation == "horizontal":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            elif orientation == "vertical":
                fig = plt.figure(figsize=(figsize[1], figsize[0]))
                ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        if plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        if not elimit:
            elimit = [dos.energies[0], dos.energies[1]]
        cond1 = dos.energies > elimit[0]
        cond2 = dos.energies < elimit[1]
        cond = np.all([cond1, cond2], axis=0)

        dE = dos.energies[1] - dos.energies[0]
        for ispin in spins:
            x = []
            y_total = []
            bar_color = []
            for idos in range(len(dos.energies[cond])):
                x.append(dos.energies[cond][idos])
                y = dos.dos[cond, ispin + 1][idos]
                y_total.append(dos_total.dos[cond, ispin + 1][idos])
                y_total_projected = dos_total_projected.dos[cond, ispin + 1][idos]
                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    y_total[-1] *= -1
                    y_total_projected *= -1
                bar_color.append(cmap(y / (y_total_projected * (vmax - vmin))))
            if orientation == "horizontal":
                #                ax.bar(x,y_total,dE,color=bar_color)
                for idos in range(len(x) - 1):
                    ax.fill_between(
                        [x[idos], x[idos + 1]],
                        [y_total[idos], y_total[idos + 1]],
                        color=bar_color[idos],
                    )
            elif orientation == "vertical":
                for idos in range(len(x) - 1):
                    ax.fill_betweenx(
                        [x[idos], x[idos + 1]],
                        [y_total[idos], y_total[idos + 1]],
                        color=bar_color[idos],
                    )
        #                ax.barh(y_total,x,dE,color=bar_color)

        return fig, ax

    def plot_stack_species(
        self,
        spins=None,
        orbitals=None,
        markersize=0.02,
        marker="o",
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
                (1, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 1),
                (192 / 255, 192 / 255, 192 / 255),
                (128 / 255, 128 / 255, 128 / 255),
                (128 / 255, 0, 0),
                (128 / 255, 128 / 255, 0),
                (0, 128 / 255, 0),
                (128 / 255, 0, 128 / 255),
                (0, 128 / 255, 128 / 255),
                (0, 0, 128 / 255),
            ]
        #        if ax is None:
        #            if orientation == 'horizontal':
        #                fig = plt.figure(figsize=figsize)
        #                ax = fig.add_subplot(111)
        #            elif orientation == 'vertical':
        #                fig = plt.figure(figsize=(figsize[1], figsize[0]))
        #                ax = fig.add_subplot(111)
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
            elimit = [-2, 2]
        dos_projected_total = self.VaspXML.dos_parametric(spin=spins, orbitals=orbitals)
        if self.VaspXML.dos_projected[0].ncols == (1 + 3 + 5) * 2:
            all_orbitals = "spd"
        elif self.VaspXML.dos_projected[0].ncols == (1 + 3 + 5 + 7) + 1:
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
        dos_total = self.VaspXML.dos_total

        cond1 = self.VaspXML.dos_total.energies > elimit[0]
        cond2 = self.VaspXML.dos_total.energies < elimit[1]
        cond = np.all([cond1, cond2], axis=0)

        for ispin in spins:
            bottom = np.zeros_like(self.VaspXML.dos_total.energies[cond])
            for ispc in range(len(self.VaspXML.species)):
                idx = (
                    np.array(self.VaspXML.initial_structure.symbols)
                    == self.VaspXML.species[ispc]
                )
                atoms = list(np.where(idx)[0])
                dos = self.VaspXML.dos_parametric(
                    atoms=atoms, spin=spins, orbitals=orbitals
                )

                x = dos.energies[cond]
                y = (
                    dos.dos[cond, ispin + 1] * dos_total.dos[cond, ispin + 1]
                ) / dos_projected_total.dos[cond, ispin + 1]

                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    if orientation == "horizontal":
                        ax.fill_between(x, bottom + y, bottom, color=colors[ispc])
                    elif orientation == "vertical":
                        ax.fill_betweenx(x, bottom + y, bottom, color=colors[ispc])
                else:
                    if orientation == "horizontal":
                        ax.fill_between(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[ispc],
                            label=self.VaspXML.species[ispc] + label,
                        )
                    elif orientation == "vertical":
                        ax.fill_betweenx(
                            x,
                            bottom + y,
                            bottom,
                            color=colors[ispc],
                            label=self.VaspXML.species[ispc] + label,
                        )
                bottom += y
        return fig, ax

    def plot_stack_orbitals(
        self,
        spins=None,
        atoms=None,
        markersize=0.02,
        marker="o",
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
                (1, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 1),
                (192 / 255, 192 / 255, 192 / 255),
                (128 / 255, 128 / 255, 128 / 255),
                (128 / 255, 0, 0),
                (128 / 255, 128 / 255, 0),
                (0, 128 / 255, 0),
                (128 / 255, 0, 128 / 255),
                (0, 128 / 255, 128 / 255),
                (0, 0, 128 / 255),
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
        atom_names = ""
        if atoms:
            print(
                "The plot only considers atoms",
                np.array(self.VaspXML.initial_structure.symbols)[atoms],
            )
            atom_names = ""
            for ispc in np.unique(
                np.array(self.VaspXML.initial_structure.symbols)[atoms]
            ):
                atom_names += ispc + "-"
        all_atoms = ""
        for ispc in np.unique(np.array(self.VaspXML.initial_structure.symbols)):
            all_atoms += ispc + "-"
        if atom_names == all_atoms:
            atom_names = ""
        dos_total = self.VaspXML.dos_total
        dos_projected_total = self.VaspXML.dos_parametric()

        if not elimit:
            elimit = [-2, 2]
        dos_projected_total = self.VaspXML.dos_parametric()

        cond1 = self.VaspXML.dos_total.energies > elimit[0]
        cond2 = self.VaspXML.dos_total.energies < elimit[1]
        cond = np.all([cond1, cond2], axis=0)
        orb_names = ["s", "p", "d"]
        orb_l = [[0], [1, 2, 3], [4, 5, 6, 7, 8]]
        for ispin in spins:
            bottom = np.zeros_like(self.VaspXML.dos_total.energies[cond])
            for iorb in range(3):
                dos = self.VaspXML.dos_parametric(
                    atoms=atoms, spin=spins, orbitals=orb_l[iorb]
                )
                x = dos.energies[cond]
                y = (
                    dos.dos[cond, ispin + 1] * dos_total.dos[cond, ispin + 1]
                ) / dos_projected_total.dos[cond, ispin + 1]
                y = np.nan_to_num(y, 0)

                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    if orientation == "horizontal":
                        ax.fill_between(x, bottom + y, bottom, color=colors[iorb])
                    elif orientation == "vertical":
                        ax.fill_betweenx(x, bottom + y, bottom, color=colors[iorb])
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
        markersize=0.02,
        marker="o",
        spin_colors=None,
        colors=None,
        elimit=None,
        figsize=(12, 6),
        ax=None,
        orientation="horizontal",
    ):

        if len(items) == 0:
            print(
                """Please provide the stacking items in which you want to plot,
                  example : {'Sr':[1,2,3],'O':[4,5,6,7,8]} will plot the stacked
                  plots of p orbitals of Sr and d orbitals of Oxygen."""
            )

        if spin_colors is None:
            spin_colors = [(0, 0, 1), (1, 0, 0)]
        src_colors = colors
        if src_colors is None:
            src_colors = [
                (1, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (1, 1, 0),
                (0, 1, 1),
                (1, 0, 1),
                (192 / 255, 192 / 255, 192 / 255),
                (128 / 255, 128 / 255, 128 / 255),
                (128 / 255, 0, 0),
                (128 / 255, 128 / 255, 0),
                (0, 128 / 255, 0),
                (128 / 255, 0, 128 / 255),
                (0, 128 / 255, 128 / 255),
                (0, 0, 128 / 255),
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
        if not elimit:
            elimit = [-2, 2]
        dos_total = self.VaspXML.dos_total

        if self.VaspXML.dos_projected[0].ncols == (1 + 3 + 5) * dos_total.ncols:
            all_orbitals = "spd"
        elif self.VaspXML.dos_projected[0].ncols == (1 + 3 + 5 + 7) * dos_total.ncols:
            all_orbitals = "spdf"
        else:
            all_orbitals = ""

        cond1 = self.VaspXML.dos_total.energies > elimit[0]
        cond2 = self.VaspXML.dos_total.energies < elimit[1]
        cond = np.all([cond1, cond2], axis=0)
        counter = 0
        colors = {}

        for ispc in items:
            colors[ispc] = src_colors[counter]
            counter += 1

        dos_projected_total = self.VaspXML.dos_parametric(spin=spins)

        for ispin in spins:
            bottom = np.zeros_like(self.VaspXML.dos_total.energies[cond])
            for ispc in items:
                idx = np.array(self.VaspXML.initial_structure.symbols) == ispc
                atoms = list(np.where(idx)[0])
                orbitals = items[ispc]

                dos = self.VaspXML.dos_parametric(
                    atoms=atoms, spin=spins, orbitals=orbitals
                )

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
                x = dos.energies[cond]

                y = (
                    dos.dos[cond, ispin + 1] * dos_total.dos[cond, ispin + 1]
                ) / dos_projected_total.dos[cond, ispin + 1]
                if ispin > 0 and len(spins) > 1:
                    y *= -1
                    if orientation == "horizontal":
                        ax.fill_between(x, bottom + y, bottom, color=colors[ispc])
                    elif orientation == "vertical":
                        ax.fill_betweenx(x, bottom + y, bottom, color=colors[ispc])
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


def plotter(
    dos, spins, markersize, marker, spin_colors, figsize, ax, orientation, labels
):

    if orientation == "horizontal":
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        x = dos.energies

        for iy in spins:
            y = dos.dos[:, iy + 1]
            if iy > 0 and len(spins) > 1:
                y *= -1
            if not labels is None:
                ax.plot(
                    x,
                    y,
                    "r-",
                    marker=marker,
                    markersize=markersize,
                    color=spin_colors[iy],
                    label=labels[iy],
                )
            else:
                ax.plot(
                    x,
                    y,
                    "r-",
                    marker=marker,
                    markersize=markersize,
                    color=spin_colors[iy],
                )

    elif orientation == "vertical":
        if ax is None:
            fig = plt.figure(figsize=(figsize[1], figsize[0]))
            ax = fig.add_subplot(111)
        else:
            fig = plt.gca()
        y = dos.energies
        for ix in spins:
            x = dos.dos[:, ix + 1]
            if ix > 0 and len(spins) > 1:
                x *= -1
            if not labels is None:
                ax.plot(
                    x,
                    y,
                    "r-",
                    marker=marker,
                    markersize=markersize,
                    color=spin_colors[ix],
                    label=labels[ix],
                )
            else:
                ax.plot(
                    x,
                    y,
                    "r-",
                    marker=marker,
                    markersize=markersize,
                    color=spin_colors[ix],
                )
    return fig, ax
