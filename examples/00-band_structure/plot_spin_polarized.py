"""

.. _ref_plotting_spin_polarized:

Plotting spin polarized band structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting spin polarized band structures.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='spin-polarized-colinear',
                                calc_type='bands')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR,
    "examples",
    "Fe",
    "vasp",
    "spin-polarized-colinear",
    "bands",
)

spins = [0, 1]

###########################

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
# When the calculation is a spin-polarized calculation. There are few more features features bandsplot can do.
# The default settings bandsplot will plot the spin-up and spin-down bands on the same plot.
pyprocar.bandsplot(code="vasp", mode="plain", fermi=5.590136, dirname=data_dir)

###############################################################################
# The line-styles or line-colors, these may be changed in the ebs section in the :doc:'pyprocar/utils/default_settings.ini' file.
#
# The keyword spins can also be used to select which spin bands to plot
spins = [1]
pyprocar.bandsplot(
    code="vasp", mode="plain", fermi=5.590136, spins=spins, dirname=data_dir
)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#

atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0, 1]

pyprocar.bandsplot(
    code="vasp",
    mode="parametric",
    fermi=5.590136,
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
spins = [0, 1]

pyprocar.bandsplot(
    code="vasp",
    mode="scatter",
    fermi=5.590136,
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)
