"""

.. _ref_plotting_colinear_bands:

Plotting band structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting band structure example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='bands')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-spin-polarized", "bands"
)

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++

pyprocar.bandsplot(code="vasp", mode="plain", fermi=5.599480, dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#
#
#
atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.bandsplot(
    code="vasp",
    mode="parametric",
    fermi=5.599480,
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)

###############################################################################
# parametric_linemode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#
#
#
atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.bandsplot(
    code="vasp",
    mode="parametric",
    fermi=5.599480,
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)


###############################################################################
# Scatter mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#
atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.bandsplot(
    code="vasp",
    mode="scatter",
    fermi=5.599480,
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)


###############################################################################
# overlay_species mode
# +++++++++++++++++++++++++++++++++++++++
#
#
#
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.bandsplot(
    code="vasp",
    mode="overlay_species",
    fermi=5.599480,
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
pyprocar.bandsplot(
    code="vasp",
    mode="overlay_orbitals",
    fermi=5.599480,
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

items = {"Fe": [4, 5, 6, 7, 8]}
pyprocar.bandsplot(
    code="vasp", mode="overlay", fermi=5.599480, items=items, dirname=data_dir
)

###############################################################################
# overlay mode by orbital names
# =============================
#
#
#

items = {"Fe": ["p", "d"]}
pyprocar.bandsplot(
    code="vasp", mode="overlay", fermi=5.599480, items=items, dirname=data_dir
)
