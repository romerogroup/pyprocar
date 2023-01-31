"""

.. _ref_plotting_plain_band_structure:

Plotting plain band structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting plain band structure .

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

pyprocar.bandsplot(
                code='qe', 
                mode='plain',
                dirname=data_dir)