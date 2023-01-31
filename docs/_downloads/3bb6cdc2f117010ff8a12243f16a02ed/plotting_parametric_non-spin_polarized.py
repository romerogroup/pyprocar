"""

.. _ref_plotting_parametric:

Plotting parametric band structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting parametric band structure for a colinear non-spin polarized calculation.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

       data_dir = pyprocar.download_example(save_dir='', 
                                    material='Fe',
                                    code='qe', 
                                    spin_calc_type='non-spin-polarized',
                                    calc_type='bands')
"""

import os
import pyprocar


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}bands{os.sep}colinear{os.sep}Fe"


atoms=[0]
orbitals=[4,5,6,7,8]
spins=[0]

pyprocar.bandsplot(
                code='qe', 
                mode='parametric',
                atoms=atoms,
                orbitaks=orbitals,
                spins=spins,
                dirname=data_dir)