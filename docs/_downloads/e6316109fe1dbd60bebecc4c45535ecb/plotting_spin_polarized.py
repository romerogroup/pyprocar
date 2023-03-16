"""

.. _ref_plotting_spin_polarized:

Plotting spin polarized band structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting spin polarized band structures.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='spin-polarized',
                                calc_type='bands')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}bands{os.sep}spin_colinear{os.sep}Fe"

spins=[0,1]

###########################

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
# When the calculation is a spin-polarized calculation. There are few more features features bandsplot can do. 
# The default settings bandsplot will plot the spin-up and spin-down bands on the same plot.
pyprocar.bandsplot(
                code='qe', 
                mode='plain',
                dirname=data_dir)

###############################################################################
# The line-styles or line-colors, these may be changed in the ebs section in the :doc:'pyprocar/utils/default_settings.ini' file.
#
# The keyword spins can also be used to select which spin bands to plot
spins = [1]
pyprocar.bandsplot(
                code='qe', 
                mode='plain',
                spins=spins,
                dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
# 

atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0,1]

pyprocar.bandsplot(
                code='qe', 
                mode='parametric',
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                vmin=0,
                vmax=1,
                dirname=data_dir)

###############################################################################
# Scatter mode
# +++++++++++++++++++++++++++++++++++++++
#
# 
#
atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0,1]

pyprocar.bandsplot(
                code='qe', 
                mode='scatter',
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                vmin=0,
                vmax=1,
                dirname=data_dir)

