"""

.. _ref_plotting_noncolinear_vasp:

Plotting non colinear band structures in VASP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting non colinear band structures in VASP.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='noncolinear',
                                calc_type='bands')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}bands{os.sep}non-colinear{os.sep}Fe"
data_dir = f"{parent_dir}{os.sep}data{os.sep}vasp{os.sep}non-colinear{os.sep}bands"

###########################

###############################################################################
# Plain mode
# +++++++++++++++
#
#
pyprocar.bandsplot(
                code='vasp', 
                mode='plain',
                dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++
#
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
# 

atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0,1,2]

pyprocar.bandsplot(
                code='vasp', 
                mode='parametric',
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                vmin=0,
                vmax=1,
                dirname=data_dir)


