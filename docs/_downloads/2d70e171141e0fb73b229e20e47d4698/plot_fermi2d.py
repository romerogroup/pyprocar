"""

.. _ref_plotting_fermi2d:

Plotting fermi2d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting fermi2d example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='spin-polarized-colinear',
                                calc_type='fermi')
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
    "fermi",
)


###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
#

pyprocar.fermi2D(code="vasp", mode="plain", fermi=5.590136, dirname=data_dir)


###############################################################################
# plain_bands mode
# +++++++++++++++++++++++++++++++++++++++
#
#

pyprocar.fermi2D(
    code="vasp", mode="plain_bands", add_legend=True, fermi=5.590136, dirname=data_dir
)


###############################################################################
# parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
#

atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0, 1]
pyprocar.fermi2D(
    code="vasp",
    mode="parametric",
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    dirname=data_dir,
    fermi=5.590136,
    spin_texture=False,
)


###############################################################################
# Selecting band indices
# +++++++++++++++++++++++++++++++++++++++
#
# You can specify specfic bands with the band indices keyword.
# band_indices will be a list of list that contain band indices to plot for a given spin. Below I only plot bands 6 and 7 for spin 0
# Also you can specify the colors of the bands as well with band_colors
band_indices = [[4, 5], []]
band_colors = [["blue", "navy"], []]
pyprocar.fermi2D(
    code="vasp",
    mode="plain_bands",
    band_indices=band_indices,
    band_colors=band_colors,
    add_legend=True,
    fermi=5.590136,
    dirname=data_dir,
)
