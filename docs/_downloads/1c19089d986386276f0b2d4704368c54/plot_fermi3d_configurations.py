"""
.. _ref_plot_fermi3d_configurations:

Plotting with Configurations in `pyprocar`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how to utilize various configurations for plotting the 3D Fermi surface using the `pyprocar` package. It provides a structured way to explore and demonstrate different configurations for the `plot_fermi_surface` function. 

Symmetry does not currently work! Make sure for Fermi surface calculations to turn off symmetry.

Preparation
-----------
Before diving into plotting, we need to download the example files. Use the following code to do this. Once downloaded, specify the `data_dir` to point to the location of the downloaded data.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='fermi')
"""

import pyvista

# You do not need this. This is to ensure an image is rendered off screen when generating example gallery.
pyvista.OFF_SCREEN = True

import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-spin-polarized", "fermi"
)

# First create the FermiHandler object, this loads the data into memory. Then you can call class methods to plot.
# Symmetry only works for specific space groups currently.
# For the actual calculations turn off symmetry and set 'apply_symmetry'=False.
fermiHandler = pyprocar.FermiHandler(code="vasp", dirname=data_dir, apply_symmetry=True)

###############################################################################

# Section 1: Plain Mode
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to plot the 3D Fermi surface using default settings.


# Section 1: Locating and Printing Configuration Files
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates where the configuration files are located in the package.
# It also shows how to print the configurations by setting print_plot_opts=True.
#

# Path to the configuration files in the package
config_path = os.path.join(pyprocar.__path__[0], "cfg")
print(f"Configuration files are located at: {config_path}")

fermiHandler.plot_fermi_surface(mode="plain", show=True, print_plot_opts=True)

###############################################################################

# Section 2: Parametric Mode with Custom Settings
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to customize the appearance of the 3D Fermi surface in parametric mode.
# We'll adjust the colormap, color limits, and other settings.

atoms = [0]
orbitals = [4, 5, 6, 7, 8]
spins = [0]
fermiHandler.plot_fermi_surface(
    mode="parametric",
    atoms=atoms,
    orbitals=orbitals,
    spins=spins,
    surface_cmap="viridis",
    surface_clim=[0, 1],
    show=True,
)
