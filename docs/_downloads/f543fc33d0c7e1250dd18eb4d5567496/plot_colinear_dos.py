"""

.. _ref_plotting_colinear_dos:

Plotting density of states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting density example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='dos')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar

data_dir = f"{pyprocar.utils.ROOT}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}qe{os.sep}non-spin-polarized{os.sep}dos"


###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++

pyprocar.dosplot(
                code='qe', 
                mode='plain',
                fermi=5.599480,
                dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
# 
#
#
atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0]

pyprocar.dosplot(
                code='qe', 
                mode='parametric',
                fermi=5.599480,
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                dirname=data_dir)

###############################################################################
# parametric_line mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
# 
#
#
atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0]

pyprocar.dosplot(
                code='qe', 
                mode='parametric_line',
                fermi=5.599480,
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                dirname=data_dir)



###############################################################################
# stack_species mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#
orbitals=[4,5,6,7,8]
spins=[0]

pyprocar.dosplot(
                code='qe', 
                mode='stack_species',
                fermi=5.599480,
                orbitals=orbitals,
                spins=spins,
                dirname=data_dir)

###############################################################################
# stack_orbtials mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#
atoms=[0]
spins=[0]
pyprocar.dosplot(
                code='qe', 
                mode='stack_orbitals',
                fermi=5.599480,
                atoms=atoms,
                spins=spins,
                dirname=data_dir)


###############################################################################
# stack mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#

items={'Fe':[4,5,6,7,8]}
pyprocar.dosplot(
                code='qe', 
                mode='stack',
                fermi=5.599480,
                items=items,
                dirname=data_dir)


###############################################################################
# overlay_species mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#
orbitals=[4,5,6,7,8]
spins=[0]

pyprocar.dosplot(
                code='qe', 
                mode='overlay_species',
                fermi=5.599480,
                orbitals=orbitals,
                spins=spins,
                dirname=data_dir)

###############################################################################
# overlay_orbtials mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#
atoms=[0]
spins=[0]
pyprocar.dosplot(
                code='qe', 
                mode='overlay_orbitals',
                fermi=5.599480,
                atoms=atoms,
                spins=spins,
                dirname=data_dir)


###############################################################################
# overlay mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#

items={'Fe':[4,5,6,7,8]}
pyprocar.dosplot(
                code='qe', 
                mode='overlay',
                fermi=5.599480,
                items=items,
                dirname=data_dir)
