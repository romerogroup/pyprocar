"""

.. _ref_plotting_parametric:

Plotting parametric band structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting parametric band structure .

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

       data_dir = pyprocar.download_example(save_dir='', 
                                    material='Fe',
                                    code='qe', 
                                    spin_calc_type='non-spin-polarized',
                                    calc_type='bands')
"""

import os
import pyprocar


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}bands{os.sep}colinear{os.sep}Fe"

###############################################################################
# Parametric mode 
# +++++++++++++++
# Parametric mode will plot the atomic, orbital, and spin projections onto the bands.
# 
# 
# ==================
# Spin projection
# ==================
#
# For collinear spin polarized and non-collinear spin calculations of DFT codes, PyProcar is able to plot the bands considering spin density (magnitude), spin magnetization and spin channels separately.
#
# For non-collinear spin calculations, ``spin=[0]`` plots the spin density (magnitude) and ``spin=[1,2,3]`` corresponds to spins oriented in :math:`S_x`, :math:`S_y` and :math:`S_z` directions respectively. 
# For parametric plots such as spin, atom and orbitals, the user should set ``mode=`parametric'``. ``cmap`` refers to the matplotlib color map used for the parametric plotting and can be modified by using the same color maps used in matplotlib.
# ``cmap='seismic'`` is recommended for parametric spin band structure plots.  For colinear spin calculations setting ``spin=[0]`` plots the spin density (magnitude) and ``spin=[1]`` plots the spin magnetization. 
#
# ==================
# Atom projection
# ==================
#
# The projection of atoms onto bands can provide information such as which atoms contribute to the electronic states near the Fermi level. 
# PyProcar counts each row of ions in the PROCAR file, starting from zero. In an example of a five atom SrVO:math:`_3`, the indexes of ``atoms`` for Sr, V and the three O atoms would be 0,1 and 2,3,4 respectively. 
# It is also possible to include more than one type of atom by using an array such as ``atoms = [0,1,3]``.
#
# =====================
# Orbital projection
# =====================
#
# The projection of atomic orbitals onto bands is also useful to identify the contribution of orbitals to bands. 
# For instance, to identify correlated :math:`d` or :math:`f` orbitals in a strongly correlated material near the Fermi level. 
# It is possible to include more than one type of orbital projection. The mapping of the index of orbitals to be used in ``orbitals`` is as follows (this is the same order from the PROCAR file). 
# Quantum Espresso, VASP and Abinit follows this order. 
# .. image:: images/orbitals.png

# In Quantum Espresso, if the calculation is a non-colinear spin-orbit calculation. THe orbitals will follow the this order:
# Creat png of the spin_orbit orbitals


atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0]

pyprocar.bandsplot(
                code='qe', 
                mode='parametric',
                atoms=atoms,
                orbitaks=orbitals,
                spins=spins,
                dirname=data_dir)