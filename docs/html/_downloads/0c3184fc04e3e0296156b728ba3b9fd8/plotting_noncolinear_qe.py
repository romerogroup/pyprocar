"""

.. _ref_plotting_noncolinear_qe:

Plotting non colinear band structures in Quantum Espresso
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting non colinear band structures in Quantum Espresso.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='non-colinear',
                                calc_type='bands')
"""
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importing pyprocar and specifying local data_dir

import os
import pyprocar


project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = f"{project_dir}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}qe{os.sep}non-colinear{os.sep}bands"

###########################

##############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++


pyprocar.bandsplot(
                code='qe', 
                mode='plain',
                dirname=data_dir)

###############################################################################
# Parametric mode
# +++++++++++++++++++++++++++++++++++++++
# Quantum Espresso expresses the projections in the coupled basis, 
# therefore orbitals takes different meanings.
# For details on the meaning of the indices of the atomic projection please refer to the user guide :ref:'atomic_projections'
#

atoms=[0]
spins=[0,1,2]
orbitals=[8,9,10,11,12,13,14,15,16,17]

pyprocar.bandsplot(
                code='qe', 
                mode='parametric',
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                vmin=0,
                vmax=1,
                dirname=data_dir)


