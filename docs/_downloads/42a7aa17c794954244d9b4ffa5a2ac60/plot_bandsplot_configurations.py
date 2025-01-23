"""
.. _ref_plot_bandsplot_configuration:

Plotting with Configurations in `pyprocar`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how to utilize various configurations for plotting band structures using the `pyprocar` package. It provides a structured way to explore and demonstrate different configurations for the `bandsplot` function.

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
                                calc_type='bands'
                               )
"""

import os

import pyprocar

# Define the directory containing the example data

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR,
    "examples",
    "Fe",
    "vasp",
    "spin-polarized-colinear",
    "bands",
)
code = "vasp"
spins = [0, 1]

###############################################################################

# Section 1: Locating and Printing Configuration Files
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates where the configuration files are located in the package.
# It also shows how to print the configurations by setting print_plot_ops=True.
#

# Path to the configuration files in the package
config_path = os.path.join(pyprocar.__path__[0], "cfg")
print(f"Configuration files are located at: {config_path}")

# Print the configurations
pyprocar.bandsplot(code=code, dirname=data_dir, fermi=5.590136, print_plot_opts=True)

###############################################################################

# Section 2: Changing cmap, clim, and Fermi line properties in Parametric Mode
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to change the colormap (cmap), color limits (clim),
# and Fermi line properties (color, linestyle, and linewidth) in parametric mode.
#

pyprocar.bandsplot(
    code=code,
    dirname=data_dir,
    mode="parametric",
    fermi=5.590136,
    atoms=[0],
    orbitals=[4, 5, 6, 7, 8],
    cmap="viridis",  # Colormap
    clim=[0, 1],  # Color limits
    fermi_color="red",  # Fermi line color
    fermi_linestyle="--",  # Fermi line linestyle
    fermi_linewidth=2.0,  # Fermi line linewidth
)


###############################################################################

# Section 3: Setting Marker and Marker Size in Scatter Mode
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to set the marker style and marker size in scatter mode.
#

pyprocar.bandsplot(
    code=code,
    dirname=data_dir,
    mode="scatter",
    fermi=5.590136,
    atoms=[0],
    orbitals=[4, 5, 6, 7, 8],
    marker=["v", "o"],  # Marker style
    markersize=[10, 5],  # Marker size list for the 2 spin plots
)

###############################################################################

# Section 4: Setting the Figure Size and DPI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to set the figure size and dots per inch (DPI) for the plot.
#

pyprocar.bandsplot(
    code=code,
    dirname=data_dir,
    mode="scatter",
    fermi=5.590136,
    atoms=[0],
    orbitals=[4, 5, 6, 7, 8],
    figure_size=(10, 6),  # Figure size (width, height)
    dpi=300,  # Dots per inch
)
