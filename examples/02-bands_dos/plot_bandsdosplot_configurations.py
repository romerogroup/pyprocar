"""
.. _ref_plot_bandsdosplot_configurations:

Plotting bandsdosplot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example illustrates how to utilize various configurations for plotting both band structures and density of states (DOS) side by side using the `pyprocar` package. It provides a structured way to explore and demonstrate different configurations for the `bandsdosplot` function.

Preparation
-----------
Before diving into plotting, we need to download the example files. Use the following code to do this. Once downloaded, specify the `bands_dir` and `dos_dir` to point to the location of the downloaded data.

.. code-block::
   :caption: Downloading example

   bands_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='bands')

   dos_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='dos')
"""

import os

import pyprocar

bands_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-spin-polarized", "bands"
)
dos_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-spin-polarized", "dos"
)

###############################################################################

# Section 1: Plain Mode with Default Settings
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to plot both band structures and DOS side by side using default settings.
# The keywords that work for `bandsplot` and `dosplot` will also work in `bandsdosplot`.
# These keyword arguments can be set in `bands_settings` and `dos_settings` as demonstrated below.
#


bands_settings = {
    "mode": "plain",
    "fermi": 5.599480,  # This will overide the default fermi value found in bands directory
    "dirname": bands_dir,
}

dos_settings = {
    "mode": "plain",
    "fermi": 5.599480,  # This will overide the default fermi value found in dos directory
    "dirname": dos_dir,
}

pyprocar.bandsdosplot(
    code="vasp",
    bands_settings=bands_settings,
    dos_settings=dos_settings,
)

###############################################################################

# Section 2: Customizing Bands and DOS Plots
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to customize the appearance of both the band structures and DOS plots.
# We'll adjust the colormap, color limits, Fermi line properties, and other settings.
#

bands_settings = {
    "mode": "scatter",
    "dirname": bands_dir,
    "fermi": 5.599480,  # This will overide the default fermi value found in bands directory
    "atoms": [0],
    "orbitals": [4, 5, 6, 7, 8],
    "cmap": "viridis",
    "clim": [0, 1],
    "fermi_color": "red",
    "fermi_linestyle": "--",
    "fermi_linewidth": 2.0,
}

dos_settings = {
    "mode": "parametric",
    "dirname": dos_dir,
    "fermi": 5.599480,  # This will overide the default fermi value found in dos directory
    "atoms": [0],
    "orbitals": [4, 5, 6, 7, 8],
    "cmap": "viridis",
    "clim": [0, 1],
    "marker": ["v", "o"],
    "markersize": [10, 5],
}

pyprocar.bandsdosplot(
    code="vasp",
    bands_settings=bands_settings,
    dos_settings=dos_settings,
)

###############################################################################

# Section 3: Adjusting Figure Size and DPI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This section demonstrates how to adjust the overall figure size and dots per inch (DPI) for the combined plot.
#

bands_settings = {"mode": "scatter", "dirname": bands_dir}

dos_settings = {"mode": "parametric", "dirname": dos_dir}

pyprocar.bandsdosplot(
    code="vasp",
    bands_settings=bands_settings,
    dos_settings=dos_settings,
    figure_size=(12, 7),
    dpi=300,
)
