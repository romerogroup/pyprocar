"""

.. _ref_plotting_de_hass_van_alphen:

Showing how to get van alphen fequencies from the fermi surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Van alphen fequencies example. De has van alphen frequencies (F) in terms of extremal fermi surface areas (A) is given below.
To compare the theoretical freuqencies we will compare with the results taken from the experimental paper
"The Fermi surfaces of copper, silver and gold. I. The de Haas-Van alphen effect"(https://doi.org/10.1098/rsta.1962.0011).


.. math::

   F = \\frac{ c \hbar A }{ 2 \pi e  }   !(cgs)

   e = 4.768e^{-10} !statcoulombs

   c = 3.0e^{10} !cm/s

   \\hbar = 1.0546e^{-27} !erg*s

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Au',
                                code='vasp', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='fermi')
"""

# sphinx_gallery_thumbnail_number = 1

###############################################################################

import pyvista
# You do not need this. This is to ensure an image is rendered off screen when generating exmaple gallery.
pyvista.OFF_SCREEN = True

###############################################################################
# importing pyprocar and specifying local data_dir

import os
import pyprocar

project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_dir = f"{project_dir}{os.sep}data{os.sep}examples{os.sep}Au{os.sep}vasp{os.sep}non-spin-polarized{os.sep}fermi"


# First create the FermiHandler object, this loads the data into memory. Then you can call class methods to plot
fermiHandler = pyprocar.FermiHandler(
                                    code="vasp",
                                    dirname=data_dir,
                                    apply_symmetry=True)



###############################################################################
# Maximal cross sectional area along the (0,0,1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 




fermiHandler.plot_fermi_cross_section_box_widget(
                            show_cross_section_area=True,
                            bands=[5],
                            transparent_mesh=True,
                            slice_normal=(0,0,1),
                            slice_origin=(0,0,0),
                            line_width=5.0,
                            mode="parametric",
                            show=True)

###############################################################################
# In the above figure we can see the cross section area is :math:`A = 4.1586 Ang^{-2} = 4.1586e^{16} cm^{-2} (cgs)`.
#
# :math:`F = \frac{ c \hbar A }{ 2 \pi e  } = 4.365e^8 G`
# 
# :math:`F_{exp} = 4.50e^7 G`

###############################################################################
# Minimal cross sectional area along the (0,0,1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 
# 

fermiHandler.plot_fermi_cross_section_box_widget(
                                show_cross_section_area=True,
                                bands=[5],
                                transparent_mesh=True,
                                slice_normal=(0,0,1),
                                slice_origin=(0,0,1.25),
                                line_width=5.0,
                                mode="parametric",
                                show=True,)

###############################################################################
# In the above figure we can see the cross section area is :math:`A = 0.1596 Ang^{-2} = 0.1596e^{16} cm^{-2} (cgs)`.
#
# :math:`F = \frac{ c \hbar A }{ 2 \pi e  } = 1.68e^7 G`
# 
# :math:`F_{exp} = 1.50e^7 G`
#

###############################################################################
# Extremal cross sectional area along the (0,1,1)
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 
# 

fermiHandler.plot_fermi_cross_section_box_widget(
                                show_cross_section_area=True,
                                bands=[5],
                                transparent_mesh=True,
                                slice_normal=(0,1,1),
                                slice_origin=(0,0,0),
                                line_width=5.0,
                                mode="parametric",
                                show=True,)


###############################################################################
# In the above figure we can see the cross section area is :math:`A = 4.3956 Ang^{-2} = 4.3956e^{16} cm^{-2} (cgs)`.
#
# :math:`F = \frac{ c \hbar A }{ 2 \pi e  } = 4.61e^8 G`
# 
# :math:`F_{exp} = 4.85e^8 G`
