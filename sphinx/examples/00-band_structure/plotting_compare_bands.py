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
                                code='qe', 
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

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
vasp_data_dir = f"{parent_dir}{os.sep}data{os.sep}vasp{os.sep}non-spin-polarized{os.sep}Fe{os.sep}bands"
qe_data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}bands{os.sep}colinear{os.sep}Fe"


###############################################################################
# When show is equal to False, bandsplot will return a EBSPlot object. 
# This object has information about the band structure and has matplotlib.axes.Axes object as an attribute.
#

ebs_plot = pyprocar.bandsplot(code='vasp', dirname = vasp_data_dir, mode='parametric', elimit=[-5,5], orbitals=[4,5,6,7,8], show=False)
pyprocar.bandsplot(code='qe', dirname = qe_data_dir, mode='plain', elimit=[-5,5], color='k',ax=ebs_plot.ax, show =True)