# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 20:49:03 2020

@author: Pedram Tavadze, Logan Lang
"""
from scipy.interpolate import CubicSpline
import numpy as np


class DensityOfStates:
    def __init__(self,
                 energies=None,
                 total=None,
                 projected=None,
                 interpolation_factor=None):
        """
        A class that contains density of states calcuated by the a density
        functional theory calculation.

        Parameters
        ----------
        energies : list float, optional
            List of points on energy spectrum . The default is None.
        total : list float, optional
            List of densities at each point. The default is None.
        projected : list float, optional
            dictionary by the following order
            projected[iatom][iprincipal][iorbital][ispin].
            ``iprincipal`` works like the principal quantum number n. The last
            index should be the total. (iprincipal = -1)
            n = iprincipal => 0, 1, 2, 3, -1 => s, p, d, total
            ``iorbital`` works similar to angular quantum number l, but not the
            same. iorbital follows this order
            (0,1,2,3,4,5,6,7,8) => s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2.
            ``ispin`` works as magnetic quantum number.
            m = 0,1, for spin up and down
            The default is None.


        interpolation_factor : int, optional
            The number of density of states points will increase by this factor
            in the interpolation.
        Returns
        -------
        None.

        """

        self.energies = energies
        self.total = total
        self.projected = projected

        if interpolation_factor is not None:
            interpolated = []
            for ispin in range(len(self.total)):
                new_energy, new_total = interpolate(
                    self.energies,
                    self.total[ispin],
                    factor=interpolation_factor)
                interpolated.append(new_total)

            self.total = interpolated

            for iatom in range(len(projected)):
                for iprincipal in range(len(projected[iatom])):
                    for iorbital in range(len(projected[iatom][iprincipal])):
                        for ispin in range(
                                len(projected[iatom][iprincipal][iorbital])):
                            x = energies
                            y = projected[iatom][iprincipal][iorbital][ispin]
                            xs, ys = interpolate(x,
                                                 y,
                                                 factor=interpolation_factor)

                            self.projected[iatom][iprincipal][iorbital][
                                ispin] = ys

            self.energies = xs

        self.ndos = len(self.energies)
        self.total = np.array(self.total)

    def dos_sum(self,
                atoms=None,
                principal_q_numbers=[-1],
                orbitals=None,
                spins=None):
        """
        +-------+-----+------+------+------+------+------+------+------+------+
        |n-lm   |  0  |   1  |  2   |   3  |   4  |   5  |   6  |   7  |   8  |
        +=======+=====+======+======+======+======+======+======+======+======+
        |-1(tot)|  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
        +=======+=====+======+======+======+======+======+======+======+======+
        |   0   |  s  |      |      |      |      |      |      |      |      |
        +=======+=====+======+======+======+======+======+======+======+======+
        |   1   |  s  |  py  |  pz  |  px  |      |      |      |      |      |
        +=======+=====+======+======+======+======+======+======+======+======+
        |   2   |  s  |  py  |  pz  |  px  | dxy  | dyz  | dz2  | dxz  |x2-y2 |
        +=======+=====+======+======+======+======+======+======+======+======+
        |  ...  | ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        +=======+=====+======+======+======+======+======+======+======+======+


        Parameters
        ----------
        atoms : list int, optional
             list of atom index needed to be sumed over.
             count from zero with the same
             order as input. The default is None.
        principal_q_numbers : list int, optional
            list of . The default is [-1].
        orbitals : TYPE, optional
            DESCRIPTION. The default is None.
        spins : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        ret : list float
            .

        """

        projected = self.projected
        principal_q_numbers = np.array(principal_q_numbers)
        if atoms is None:
            atoms = np.arange(len(projected), dtype=int)
        if spins is None:
            spins = np.arange(len(projected[0][0][0]), dtype=int)

        if orbitals is None:
            orbitals = np.arange(len(projected[0][0]), dtype=int)
        orbitals = np.array(orbitals)

        ret = np.zeros(shape=(2, self.ndos))
        for iatom in atoms:
            for iprinc in principal_q_numbers:
                for ispin in spins:
                    temp = np.array(projected[iatom][iprinc])

                    ret[ispin, :] += temp[orbitals, ispin].sum(axis=0)

        return ret


def interpolate(x, y, factor=2):
    """
    Interplates the function y=f(x) by increasing the x points by the factor.

    Parameters
    ----------
    x : list float
        points of x.
    y : list float
        points of y=f(x).
    factor : int, optional
        Multiplies the number of x points by this factor. The default is 2.

    Returns
    -------
    xs : list float
        points in which y was interpolated.
    ys : list float
        interpolated points.

    """

    cs = CubicSpline(x, y)
    xs = np.linspace(min(x), max(x), len(x) * factor)
    ys = cs(xs)

    return xs, ys


# =======
# Created on Mon Jun 15 21:35:50 2020

# @author: lllan
# """

# import os
# import sys
# import numpy as np
# from pychemia import HAS_MATPLOTLIB

# class DensityOfStates:
#     """
#     Stores the density of states
#     It its basically a numpy array with the first column representing energies
#     and all the extra columns representing density of states.
#     Those several columns could contain partial contributions due to spin and or
#     orbital
#     """

#     def __init__(self, table=None, title=None,labels=None):
#         self._dos = None
#         #self._principle_qnumber = None
#         self._min_energy = None
#         self._max_energy = None
#         self._max_dos = None
#         self.title = title
#         self.labels = labels
#         self.ncols = 1

#         if table is not None:
#             self._dos = np.array(table)
#             self.ncols = self._dos.shape[1] - 1
#             self._min_energy = min(table[:, 0])
#             self._max_energy = max(table[:, 0])
#             self._max_dos = max(table[:, 1])

#         if self.labels == None:
#             self.labels = np.arange(self.ncols)

#     @staticmethod
#     def read(filename, title=None):
#         """
#         Reads a file and returns a DensityOFStates object.
#         The file could contain concatenated two columns representing
#         the desnity of states. However, the energies should be
#         repetitivive and the number of values be the same for all the
#         sets
#         :param filename: (string) the file that contains the density of states
#         :param title: (string) customize title
#         :return: (DensityOfStates) object
#         """
#         table = np.loadtxt(filename)
#         table = np.array(table)
#         if title is None:
#             root, ext = os.path.splitext(os.path.basename(filename))
#             name = root
#         else:
#             name = title

#         jump = 0
#         jumplist = [0]
#         nsets = 1
#         for iline in range(len(table) - 1):
#             if table[iline, 0] != table[iline - jump, 0]:
#                 print(iline, jump, table[iline, 0], table[iline - jump, 0])
#                 raise ValueError("No consistency on energy values")
#             if table[iline + 1, 0] < table[iline, 0]:
#                 jump = iline + 1
#                 jumplist.append(jump)
#                 nsets += 1

#         if nsets > 1:
#             jump1 = jumplist[1] - jumplist[0]
#             for i in range(1, nsets - 1):
#                 if jumplist[i + 1] - jumplist[i] != jump1:
#                     raise ValueError("No equal jumps")
#             table2 = np.zeros((jump1, nsets + 1))
#             table2[:, 0] = table[:jump1, 0]
#             for i in range(nsets):
#                 table2[:, i + 1] = table[i * jump1:(i + 1) * jump1, 1]
#                 assert (np.all(table[i * jump1:(i + 1) * jump1, 0] == table[:jump1, 0]))
#         else:
#             table2 = table

#         dos = DensityOfStates(table=table2, title=name)
#         return dos

#     @property
#     def dos(self):
#         return self._dos

#     @property
#     def energies(self):
#         """
#         The energy values
#         :return: (numpy.ndarray) One-dimensional array of energies
#         """
#         return self._dos[:, 0]

#     @property
#     def values(self):
#         """
#         The density of states values, could be multidimensional
#         :return: (numpy.ndarray) Density of states values
#         """
#         if self.ncols > 1:
#             return self._dos[:, range(1, self.ncols + 1)]
#         else:
#             return self._dos[:, 1]

#     def save_txt(self,filename=None):
#         """
#         writes the density of states in a file using numpy savetxt.
#         If filename not defined, the title of object is used as filename

#         :param filename
#         """
#         if filename == None:
#             filename = self.title.replace(' ','')+'.txt'

#         header = ('%12s'+'%12s'*self.ncols) % tuple(self.labels)
#         fmt = ('%12.3f '+'%12.4f'*self.ncols)
#         np.savetxt(fname=filename,fmt=fmt,X=self.dos,header=header)

#         #-34.460  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00

#     def to_dict(self):
#         """
#         returns the object as a python dictionary

#         :return {'labels':labels,'energies':energies,title:values}
#         """
#         return {'labels':self.labels,'energies':self.energies,self.title:self.values}

# def plot_one_dos(dosobj, ax=None, horizontal=True, figwidth=16, figheight=12):
#     """
#     Plot a single density of states, if the values contains
#     several dimensions all the dimensions are plotted
#     If the axes is no given, a new figure is created and the
#     axes is returned
#     :param figheight:
#     :param figwidth:
#     :param dosobj: (DensityOfStates) object
#     :param ax: (matplotlib.axes.Axes) object
#     :param horizontal: (bool) if the plot is horizontal or
#     :return: (matplotlib.figure.Figure, matplotlib.axes.Axes) the (fig, ax) tuple
#     """
#     if HAS_MATPLOTLIB:
#         import matplotlib.pyplot as plt
#     else:
#         raise NotImplementedError

#     if ax is None:
#         fig = plt.figure()
#         fig.set_figheight(figheight)
#         fig.set_figwidth(figwidth)
#         ax = fig.add_subplot(111)
#         if horizontal:
#             ax.set_xlabel('Energy')
#         else:
#             ax.set_ylabel('Energy')
#     else:
#         fig = plt.gcf()

#     xx = dosobj.energies

#     if dosobj.ncols > 1:
#         for i in range(dosobj.ncols):
#             yy = dosobj.values[:, i]

#             if horizontal:

#                 ax.plot(xx, yy, label=dosobj.labels[i+1])
#             else:
#                 ax.plot(yy, xx, label=dosobj.labels[i+1])

#     else:
#         yy = dosobj.values
#         if horizontal:
#             ax.plot(xx, yy)
#         else:
#             ax.plot(yy, xx)

#     # fig.savefig('test.pdf')
#     return fig, ax

# def plot_many_dos(doslist, minenergy=None, maxenergy=None, figwidth=16, figheight=12):
#     """
#     Plot multiple densities of states
#     :param doslist: (list) list of DensityOfStates objects
#     :param minenergy: (float) minimal energy to display
#     :param maxenergy: (float) maximal energy to display
#     :param figheight: (float) Height of figure
#     :param figwidth: (float) Width of figure
#     """
#     import matplotlib.pyplot as plt
#     ndos = len(doslist)
#     if minenergy is None:
#         minenergy = min([min(x.energies) for x in doslist])
#     if maxenergy is None:
#         maxenergy = max([max(x.energies) for x in doslist])
#     minval = sys.float_info.max
#     maxval = sys.float_info.min
#     for idos in doslist:
#         for i in idos.dos:
#             if minenergy < i[0] < maxenergy:
#                 for icol in range(idos.ncols):
#                     if i[icol + 1] > maxval:
#                         maxval = i[icol + 1]
#                     if i[icol + 1] < minval:
#                         minval = i[icol + 1]

#     fig, ax = plt.subplots(nrows=1, ncols=ndos, sharex=False, sharey=True, squeeze=True)
#     fig.set_figwidth(figwidth)
#     fig.set_figheight(figheight)
#     plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
#     for i in range(ndos):
#         plot_one_dos(doslist[i], ax[i], horizontal=False)
#         ax[i].set_xlim(1.1 * minval, 1.1 * maxval)
#         ax[i].set_ylim(minenergy, maxenergy)
#         ax[i].set_xlabel(doslist[i].title)
#         ax[i].spines['bottom'].set_linewidth(10)
#         ax[i].spines['left'].set_linewidth(10)
#     ax[0].set_ylabel('Energy')
#     fig.savefig('test.pdf')
#     return fig, ax
# >>>>>>> e937792a3f68b211f23cc04a4ff068e4b6ae5535
