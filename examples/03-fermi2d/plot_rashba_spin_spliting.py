"""

.. _ref_plotting_plotting_rashba_spin_spliting:

Plotting rashba spin splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting rashba spin splitting example. 
From our first paper we had an example to plot the different spin texture projections of BiSb at a constant energy surface 0.60eV above the fermei level and -0.90ev below the fermi level. 

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='BiSb_monolayer',
                                code='vasp', 
                                spin_calc_type='non-colinear',
                                calc_type='fermi')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar

data_dir = f"{pyprocar.utils.ROOT}{os.sep}data{os.sep}examples{os.sep}BiSb_monolayer{os.sep}vasp{os.sep}non-colinear{os.sep}fermi"


###############################################################################
# energy = 0.60 sx projection no arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=0.60,
               spin_texture=True,
               no_arrow=True,
               spin_projection='x',
               plot_color_bar=True)

###############################################################################
# energy = 0.60 sy projection no arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=0.60,
               spin_texture=True,
               no_arrow=True,
               spin_projection='y',
               plot_color_bar=True)


###############################################################################
# energy = 0.60 sz projection no arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=0.60,
               spin_texture=True,
               no_arrow=True,
               spin_projection='z',
               plot_color_bar=True)

###############################################################################
# energy = -0.90 sx projection no arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=-0.90,
               spin_texture=True,
               no_arrow=True,
               spin_projection='x',
               plot_color_bar=True)

###############################################################################
# energy = -0.90 sy projection no arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=-0.90,
               spin_texture=True,
               no_arrow=True,
               spin_projection='y',
               plot_color_bar=True)


###############################################################################
# energy = -0.90 sz projection no arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=-0.90,
               spin_texture=True,
               no_arrow=True,
               spin_projection='z',
               plot_color_bar=True)


###############################################################################
# energy = 0.60 sx projection with arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=0.60,
               spin_texture=True,
               spin_projection='x',
               arrow_size=3,
               arrow_density=6,
               plot_color_bar=True)

###############################################################################
# energy = -0.90 sx projection with arrows 
# +++++++++++++++++++++++++++++++++++++++++++++
#

pyprocar.fermi2D(code = 'vasp',
               dirname=data_dir,
               energy=-0.90,
               spin_texture=True,
               spin_projection='x',
               arrow_size=3,
               arrow_density=6,
               plot_color_bar=True)
