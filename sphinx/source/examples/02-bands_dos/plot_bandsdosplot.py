"""

.. _ref_plot_bandsdosplot:

Plotting bandsdosplot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting bandsdosplot example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

   bands_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='bands')

   dos_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='dos')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar

bands_dir = f"{pyprocar.utils.ROOT}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}bands"
dos_dir = f"{pyprocar.utils.ROOT}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}dos"
###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
# The keywords that works for bandsplot and dosplot will work in bandsdosplot. 
# These keyword arguments can be set in bands_settings and dos_settings as done below.
#

bands_settings = {
                  'mode':'plain',
                  'fermi':5.599480, # This will overide the default fermi value found in bands directory
                  'dirname': bands_dir
                  }

dos_settings = {
               'mode':'plain',
               'fermi':5.599480,   # This will overide the default fermi value found in dos directory
               'dirname': dos_dir
                }

pyprocar.bandsdosplot(code='vasp',
                bands_settings=bands_settings,
                dos_settings=dos_settings,
                )


