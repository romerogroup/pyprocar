"""

.. _ref_plotting_colinear_dos:

Plotting density of states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting density example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='SrVO3',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='fermi')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "SrVO3", "vasp", "non-spin-polarized", "fermi"
)


###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++

pyprocar.dosplot(code="vasp", mode="plain", fermi=5.3017, dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#
#
#
atoms = [0, 1, 2, 3, 4]
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.dosplot(
    code="vasp",
    mode="parametric",
    fermi=5.3017,
    atoms=atoms,
    orbitals=orbitals,
    clim=[0, 1],
    spins=spins,
    dirname=data_dir,
)

###############################################################################
# parametric_line mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#
#
#
atoms = [0, 1, 2, 3, 4]
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.dosplot(
    code="vasp",
    mode="parametric_line",
    fermi=5.3017,
    atoms=atoms,
    orbitals=orbitals,
    clim=[0, 1],
    spins=spins,
    dirname=data_dir,
)


###############################################################################
# stack_species mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.dosplot(
    code="vasp",
    mode="stack_species",
    fermi=5.3017,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)

###############################################################################
# stack_orbtials mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#
atoms = [2, 3, 4]
spins = [0]
pyprocar.dosplot(
    code="vasp",
    mode="stack_orbitals",
    fermi=5.3017,
    atoms=atoms,
    spins=spins,
    dirname=data_dir,
)


###############################################################################
# stack mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#

items = dict(Sr=[0], O=[1, 2, 3], V=[4, 5, 6, 7, 8])
pyprocar.dosplot(code="vasp", mode="stack", fermi=5.3017, items=items, dirname=data_dir)


###############################################################################
# overlay_species mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.dosplot(
    code="vasp",
    mode="overlay_species",
    fermi=5.3017,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)

###############################################################################
# overlay_orbtials mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#
atoms = [0]
spins = [0]
pyprocar.dosplot(
    code="vasp",
    mode="overlay_orbitals",
    fermi=5.3017,
    atoms=atoms,
    spins=spins,
    dirname=data_dir,
)


###############################################################################
# overlay mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#

items = dict(Sr=[0], O=[1, 2, 3], V=[4, 5, 6, 7, 8])
pyprocar.dosplot(
    code="vasp", mode="overlay", fermi=5.3017, items=items, dirname=data_dir
)
