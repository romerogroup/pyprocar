"""

.. _ref_plotting_noncolinear_dos_vasp:

Plotting non colinear dos in VASP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting non colinear dos in VASP.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-colinear',
                                calc_type='dos')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-colinear", "dos"
)


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

pyprocar.dosplot(
    code="vasp",
    mode="parametric",
    fermi=5.5962,
    atoms=atoms,
    orbitals=orbitals,
    clim=[0, 1],
    spins=spins,
    dirname=data_dir,
)
