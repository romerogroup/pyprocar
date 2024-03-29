"""

.. _ref_plotting_compare_bands:

Comparing band structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing band structures example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    vasp_data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='bands')

    qe_data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='bands')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
#

import os
import pyprocar

vasp_data_dir = f"{pyprocar.utils.ROOT}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}bands"
qe_data_dir = f"{pyprocar.utils.ROOT}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}qe{os.sep}non-spin-polarized{os.sep}bands"


###############################################################################
# When show is equal to False, bandsplot will return a maplotlib.Figure and maplotlib.axes.Axes object
#

fig, ax = pyprocar.bandsplot(code='vasp', dirname = vasp_data_dir, mode='parametric',fermi=5.599480, elimit=[-5,5], orbitals=[4,5,6,7,8], show=False)
pyprocar.bandsplot(code='qe', dirname = qe_data_dir, mode='plain',fermi=18.2398, elimit=[-5,5], color='k',ax=ax, show =True)