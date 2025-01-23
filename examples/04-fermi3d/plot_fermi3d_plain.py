"""

.. _ref_plotting_fermi3d_plain:

Plotting fermi3d plain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symmetry does not currently work! Make sure for fermi surface calculations turn off symmetry

Plotting fermi3d plain example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='fermi')
"""

# sphinx_gallery_thumbnail_number = 2

###############################################################################

import pyvista

# You do not need this. This is to ensure an image is rendered off screen when generating exmaple gallery.
pyvista.OFF_SCREEN = True

###############################################################################
# importing pyprocar and specifying local data_dir

import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-spin-polarized", "fermi"
)


# First create the FermiHandler object, this loads the data into memory. Then you can call class methods to plot
# Symmetry only works for specfic space groups currently.
# For the actual calculations turn off symmetry and set 'apply_symmetry'=False
fermiHandler = pyprocar.FermiHandler(code="vasp", dirname=data_dir, apply_symmetry=True)

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
#

fermiHandler.plot_fermi_surface(
    mode="plain",
    show=True,
)


###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
#

atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0]
fermiHandler.plot_fermi_surface(
    mode="parametric",
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    show=True,
)
