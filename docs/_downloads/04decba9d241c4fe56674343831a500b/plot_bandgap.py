"""

.. _ref_example_bandgap:

Example of finding the bandgap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bandgap of a calculation can be found by:

.. code-block::
   :caption: General Format

   pyprocar.bandgap(procar="PROCAR", outcar="OUTCAR", code="vasp")


NOTE:
The bandgap calculation should be done for non-self consistent (band structure) calculations. 

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='bands')
"""

# sphinx_gallery_thumbnail_number = 1


###############################################################################
# importing pyprocar and specifying local data_dir

import os

import numpy as np

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-spin-polarized", "bands"
)

band_gap = pyprocar.bandgap(dirname=data_dir, code="vasp")
