"""

.. _ref_plot_unfolding:

Unfolding Band Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unfolding Band Structure example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    supercell_dir = pyprocar.download_example(save_dir='', 
                                material='MgB2',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='supercell_bands')

    primitive_dir = pyprocar.download_example(save_dir='', 
                                material='MgB2',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='primitive_bands')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os

import numpy as np

import pyprocar

supercell_dir = os.path.join(
    pyprocar.utils.DATA_DIR,
    "examples",
    "MgB2_unfold",
    "vasp",
    "non-spin-polarized",
    "supercell_bands",
)
primitive_dir = os.path.join(
    pyprocar.utils.DATA_DIR,
    "examples",
    "MgB2_unfold",
    "vasp",
    "non-spin-polarized",
    "primitive_bands",
)


###############################################################################
# Plotting primitive bands
# +++++++++++++++++++++++++++++++++++++++

pyprocar.bandsplot(
    code="vasp", mode="plain", fermi=4.993523, elimit=[-15, 5], dirname=primitive_dir
)


###############################################################################
# Unfolding of the supercell bands
# +++++++++++++++++++++++++++++++++++++++
#
# Here we do unfolding of the supercell bands. In this calculation,
# the POSCAR and KPOINTS will be different from the primitive cell
# For the POSCAR, we create a 2 2 2 supercell from the primitive.
# For the KPOINTS, the paths need to be changed to reflect the change in the unitcell

pyprocar.unfold(
    code="vasp",
    mode="plain",
    unfold_mode="both",
    fermi=5.033090,
    dirname=supercell_dir,
    elimit=[-15, 5],
    supercell_matrix=np.diag([2, 2, 2]),
)
