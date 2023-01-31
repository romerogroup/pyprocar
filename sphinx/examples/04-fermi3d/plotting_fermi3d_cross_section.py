"""

.. _ref_plotting_fermi3d_cross_section:

Plotting fermi3d cross_section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting fermi3d cross_section example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='non-colinear',
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
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}fermi{os.sep}noncolinear{os.sep}Fe"


# First create the FermiHandler object, this loads the data into memory. Then you can call class methods to plot
fermiHandler = pyprocar.FermiHandler(
                                    code="qe",
                                    dirname=data_dir,
                                    apply_symmetry=True)




###############################################################################
# Cross section
# +++++++++++++++
# 
#

# show_cross_section_area can show the outermost cross section area
show_cross_section_area=False

# slice_normal is the initial orientation of the the cross section plane widget
slice_normal=(1,0,0)

# slice_origin is the initial position of the center of the cross section plane widget
slice_origin=(0,0,0)

# line_width is the size of the line of the cross section
line_width=5.0

# when you run this code, you will be able to adjust the widget manually. 
# If you want to save the position of the widget use this keyword argument to save an image.
# This must be a string to the filename where you will save the image
#save_2d_slice=''

fermiHandler.plot_fermi_cross_section(
                              show_cross_section_area=show_cross_section_area,
                              slice_normal=slice_normal,
                              slice_origin=slice_origin,
                              line_width=line_width,

                              mode="spin_texture",
                              spin_texture=True,
                              arrow_size=0.5,
                              show=True,)

###############################################################################
# Cross section. Save slice
# +++++++++++++++
# 
#

# show_cross_section_area can show the outermost cross section area
show_cross_section_area=True

# when you run this code, you will be able to adjust the widget manually. 
# If you want to save the position of the widget use this keyword argument to save an image.
# This must be a string to the filename where you will save the image
save_2d_slice='2d_slice.png'

fermiHandler.plot_fermi_cross_section(
                              show_cross_section_area=show_cross_section_area,
                              slice_normal=slice_normal,
                              slice_origin=slice_origin,
                              line_width=line_width,

                              mode="spin_texture",
                              spin_texture=True,
                              arrow_size=0.5,
                              save_2d_slice=save_2d_slice,
                              show=True,)


