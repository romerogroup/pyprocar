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
                                spin_calc_type='non-spin-polarized',
                                calc_type='fermi')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}fermi{os.sep}colinear{os.sep}Fe"

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
#

pyprocar.fermi2D(code = 'qe', 
                dirname=data_dir)


###############################################################################
# Projection
# +++++++++++++++++++++++++++++++++++++++
#
# Does not work. Contact developers
#

atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0]
pyprocar.fermi2D(code = 'qe', 
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                dirname=data_dir, 
                spin_texture=False)
