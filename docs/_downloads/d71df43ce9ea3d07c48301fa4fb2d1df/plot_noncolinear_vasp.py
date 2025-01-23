"""

.. _ref_plotting_noncolinear_vasp:

Plotting non colinear band structures in VASP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting non colinear band structures in VASP.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-colinear',
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
    "non-colinear",
    "bands",
)


###########################

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
#
pyprocar.bandsplot(code="vasp", mode="plain", fermi=5.596151, dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#

atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0]

pyprocar.bandsplot(
    code="vasp",
    mode="parametric",
    fermi=5.596151,
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
)
