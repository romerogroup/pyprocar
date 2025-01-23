"""
.. _ref_plot_dos_configuration:

Plotting with Configurations in `pyprocar`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how to utilize various configurations for plotting the density of states (DOS) using the `pyprocar` package. It provides a structured way to explore and demonstrate different configurations for the `dosplot` function.

Preparation
-----------
Before diving into plotting, we need to download the example files. Use the following code to do this. Once downloaded, specify the `data_dir` to point to the location of the downloaded data.

.. code-block::
   :caption: Downloading example

    import pyprocar

    data_dir = pyprocar.download_example(
                                save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='spin-polarized-colinear',
                                calc_type='dos'
                               )
"""

import os

import pyprocar

# Define the directory containing the example data
code = "vasp"
data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", code, "spin-polarized-colinear", "dos"
)

spins = [0, 1]

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
pyprocar.dosplot(code=code, dirname=data_dir, fermi=5.599480, print_plot_opts=True)

###############################################################################

# Section 2: Changing cmap, clim, and Fermi line properties
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to change the colormap (cmap), color limits (clim),
# and Fermi line properties (color, linestyle, and linewidth).
#

pyprocar.dosplot(
    code=code,
    dirname=data_dir,
    fermi=5.599480,
    atoms=[0],
    orbitals=[4, 5, 6, 7, 8],
    mode="parametric",
    cmap="viridis",  # Colormap
    clim=[0, 1],  # Color limits
    fermi_color="red",  # Fermi line color
    fermi_linestyle="--",  # Fermi line linestyle
    fermi_linewidth=2.0,  # Fermi line linewidth
)

###############################################################################

# Section 4: Setting the Figure Size and DPI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to set the figure size and dots per inch (DPI) for the plot.
#

pyprocar.dosplot(
    code=code,
    dirname=data_dir,
    fermi=5.599480,
    atoms=[0],
    orbitals=[4, 5, 6, 7, 8],
    mode="parametric_line",
    clim=[0, 1],
    figure_size=(10, 6),  # Figure size (width, height)
    dpi=300,  # Dots per inch
    grid=True,  # Boolean for grid
)
