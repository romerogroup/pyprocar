"""
.. _ref_plotting_fermi2d_configurations:

Plotting with Configurations in `pyprocar`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how to utilize various configurations for plotting the 2D Fermi surface with non-colinear spin textures using the `pyprocar` package. It provides a structured way to explore and demonstrate different configurations for the `fermi2D` function. For more information about `fermi2D`, refer to :ref:`fermi2d`.

Preparation
-----------
Before diving into plotting, we need to download the example files. Use the following code to do this. Once downloaded, specify the `data_dir` to point to the location of the downloaded data.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-colinear',
                                calc_type='fermi')
"""

import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-colinear", "fermi"
)

###############################################################################

# Section 1: Locating and Printing Configuration Files
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates where the configuration files are located in the package.
# It also shows how to print the configurations by setting print_plot_opts=True.
#

# Path to the configuration files in the package
config_path = os.path.join(pyprocar.__path__[0], "cfg")
print(f"Configuration files are located at: {config_path}")

# Print the configurations
pyprocar.fermi2D(code="vasp", dirname=data_dir, fermi=5.599480, print_plot_opts=True)

###############################################################################

# Section 2: Spin Texture Projection with Custom Settings
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to customize the appearance of the spin texture arrows.
# We'll adjust the colormap, color limits.
#

pyprocar.fermi2D(
    code="vasp",
    fermi=5.599480,
    dirname=data_dir,
    spin_texture=True,
    spin_projection="x",
    arrow_size=3,
    arrow_density=10,
    plot_color_bar=True,
    cmap="jet",
    clim=[0, 1],
)


###############################################################################

# Section 3: Adjusting DPI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to adjust the dots per inch (DPI) for the combined plot.
#


pyprocar.fermi2D(
    code="vasp",
    dirname=data_dir,
    fermi=5.599480,
    spin_texture=True,
    spin_projection="z",
    arrow_size=3,
    arrow_density=10,
    plot_color_bar=True,
    cmap="jet",
    clim=[0, 1],
    dpi=300,
)
