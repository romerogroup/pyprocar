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
                                code='qe', 
                                spin_calc_type='spin-polarized-colinear',
                                calc_type='fermi')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar


project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = f"{project_dir}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}qe{os.sep}spin-polarized-colinear{os.sep}fermi"


###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
#

pyprocar.fermi2D(code = 'qe', 
               mode='plain',
               dirname=data_dir)


###############################################################################
# plain_bands mode
# +++++++++++++++++++++++++++++++++++++++
#
#

pyprocar.fermi2D(code = 'qe', 
               mode='plain_bands',
               add_legend=True,
               dirname=data_dir)





###############################################################################
# parametric mode
# +++++++++++++++++++++++++++++++++++++++
#
# Does not work. Contact developers
#

atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0,1]
pyprocar.fermi2D(code = 'qe',
               mode='parametric', 
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                dirname=data_dir, 
                spin_texture=False)



###############################################################################
# Selecting band indices
# +++++++++++++++++++++++++++++++++++++++
#
# You can specify specfic bands with the band indices keyword. 
# band_indices will be a list of list that contain band indices to plot for a given spin. Below I only plot bands 6 and 7 for spin 0
# Also you can specify the colors of the bands as well with band_colors
band_indices = [[6,7],[]]
band_colors = [['blue','navy'], []]
pyprocar.fermi2D(code = 'qe', 
               mode='plain_bands',
               band_indices = band_indices,
               band_colors=band_colors,
               add_legend=True,
               dirname=data_dir)