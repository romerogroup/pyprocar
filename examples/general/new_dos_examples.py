import copy
import logging
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.signal import find_peaks

import pyprocar

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)


from dotenv import load_dotenv

load_dotenv()

import os

print(os.getenv("DATA_DIR"))
DATA_DIR = Path(os.getenv("DATA_DIR"))


NON_SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "bands" / "non-spin-polarized"
SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "bands" / "spin-polarized"
NON_COLINEAR_DIR = DATA_DIR / "examples" / "bands" / "non-colinear"


DOS_NON_SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "dos" / "non-spin-polarized"
DOS_SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "dos" / "spin-polarized"
DOS_NON_COLINEAR_DIR = DATA_DIR / "examples" / "dos" / "non-colinear"


GAMMA_POINT_DIR = DATA_DIR / "examples" / "bands" / "atomic_levels" / "hBN-C2"


from pyprocar.core.dos import DensityOfStates
from pyprocar.plotter.dos_plot import DOSPlotter


def test_plot_horizontal_total_line():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)
    total = dos_non_spin_polarized.total
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total)
    plotter.show()
    
def test_plot_horizontal_projected_sum_line():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)
    atoms = [0, 1]
    orbitals = [3,4,5,6,7,8]
    
    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0])
    plotter = DOSPlotter(orientation="horizontal")
    
    plotter.plot(total)
    plotter.plot(projected_sum)
    plotter.legend()
    plotter.show()

def test_plot_horizontal_projected_sum_line_integral_normalized():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="integral")
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(projected_sum)
    plotter.show()


def test_plot_horizontal_total_with_projected_sum_scalars_line():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="total_projection")

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()
    
def test_plot_horizontal_total_with_projected_sum_scalars_line_flip_channel_mode_per_channel_colorbar():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, 
                                                                 orbitals=orbitals, 
                                                                 spins=[0,1], 
                                                                 norm_mode="total_projection")

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, 
                 scalars_mode="line", 
                 channel_mode="flip", 
                 scalars_show_colorbar="per_channel")
    plotter.show()
    
    
def test_plot_horizontal_projected_sum_with_grouped_kwargs():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]

    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0,1])

    total = dos_non_spin_polarized.total
    plotter = DOSPlotter(orientation="horizontal")
    plot_kwargs = [
        {"linewidth": 1.0, "alpha": 0.5},
        {"linewidth": 2.0, "alpha": 1.0}
    ]
    plotter.plot(projected_sum, plot_kwargs=plot_kwargs)
    plotter.plot(total)
    
    plotter.legend()
    plotter.show()
    

    
def test_plot_horizontal_total_with_projected_sum_scalars_line_with_grouped_kwargs():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0,1], norm_mode="total_projection")

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line", linewidth=[1.0, 2.0], alpha=[0.5, 1.0])
    plotter.show()
    
    
def test_plot_horizontal_total_with_projected_sum_scalars_line_flip_channel_mode():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0,1], norm_mode="total_projection")

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line", 
                 linewidth=[1.0, 2.0], 
                 alpha=[0.5, 1.0],
                 channel_mode="flip")
    plotter.show()
    
def test_plot_horizontal_total_with_projected_sum_scalars_fill_flip_channel_mode():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0,1], norm_mode="total_projection")

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, 
                 scalars_mode="fill", 
                #  linewidth=[1.0, 2.0], 
                 alpha=[0.5, 1.0],
                 channel_mode="flip")
    plotter.show()
    
    
    
    
    
def test_plot_horizontal_total_with_projected_sum_scalars_fill_with_grouped_kwargs():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0,1], norm_mode="total_projection")

    plotter = DOSPlotter(orientation="horizontal")
    plot_kwargs = [
        { "alpha": 0.5},
        { "alpha": 1.0}
    ]
    
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="fill", plot_kwargs = plot_kwargs)
    plotter.show()
    
def test_plot_horizontal_total_with_projected_sum_scalars_fill():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)

    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="total_projection")
    
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="fill")
    plotter.show()
    

def test_plot_vertical_total_with_projected_sum_scalars_line():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)

    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="total_projection")
    
    
    plotter = DOSPlotter(orientation="vertical")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()
    
def test_plot_vertical_total_with_projected_sum_scalars_fill():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)

    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="total_projection")
    
    plotter = DOSPlotter(orientation="vertical")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="fill")
    plotter.show()
    
    
def test_non_spin_polarized_total_with_gradients_line(**kwargs):
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    total_gradient = total.compute_gradient_property(order=1)

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, vectors_data=total_gradient, channel_mode="flip", **kwargs)
    plotter.show() 
    
    
#--------------------------------------------------------
# Spin polarized testing
#--------------------------------------------------------

def test_spin_polarized_plot_total_with_projected_sum_scalars_line():
    dos_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_spin_polarized.total
    projected_sum = dos_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="total_projection")
    
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()


def test_non_colinear_plot_total_with_projected_sum_scalars_line():
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    projected_sum = dos_non_colinear.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], norm_mode="total_projection")
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()

###########################################################
# Magnetization testing
###########################################################
def test_non_colinear_plot_total_with_magnetization_scalars_line():
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    magnetization = dos_non_colinear.compute_magnetization(atoms=atoms, orbitals=orbitals)
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=magnetization, scalars_mode="line")
    plotter.show()
    
    
    
def test_non_colinear_plot_total_with_spin_texture_norm_mode_magnetization_scalars_line():
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    magnetization = dos_non_colinear.compute_magnetization(atoms=atoms, 
                                                           orbitals=orbitals, 
                                                           norm_mode="magnetization")
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=magnetization, scalars_mode="line")
    plotter.show()
    
    
    
def test_non_colinear_plot_total_with_mag_norm_mode_magnetization_from_total_line():
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    magnetization = dos_non_colinear.compute_magnetization(atoms=atoms, 
                                                           orbitals=orbitals, 
                                                           norm_mode="magnetization", 
                                                           from_total=True, 
                                                           fill_value=None)
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=magnetization, scalars_mode="line")
    plotter.show()

###########################################################
# Spin texture magnitude testing
###########################################################
def test_non_colinear_plot_total_with_spin_texture_magnitude_scalars_line(**kwargs):
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    spin_texture_magnitude = dos_non_colinear.compute_spin_texture_magnitude(atoms=atoms, orbitals=orbitals, **kwargs)
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=spin_texture_magnitude, scalars_mode="line")
    plotter.show()    
    
def test_non_colinear_plot_total_with_spin_mag_norm_mode_spin_texture_magnitude_scalars_line(**kwargs):
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    spin_texture_magnitude = dos_non_colinear.compute_spin_texture_magnitude(atoms=atoms, orbitals=orbitals, norm_mode="spin_magnitude", **kwargs)
    
    plotter = DOSPlotter(orientation="horizontal")
    array = spin_texture_magnitude.to_array()
    plotter.plot(total, scalars_data=spin_texture_magnitude, scalars_mode="line")
    plotter.show() 
    
def test_non_colinear_plot_total_with_mag_norm_mode_spin_texture_magnitude_scalars_line(**kwargs):
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    spin_texture_magnitude = dos_non_colinear.compute_spin_texture_magnitude(atoms=atoms, orbitals=orbitals, norm_mode="magnetization", **kwargs)

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=spin_texture_magnitude, scalars_mode="line")
    plotter.show()
    
def test_non_colinear_plot_total_with_spin_mag_norm_mode_spin_texture_magnitude_from_total_scalars_line(**kwargs):
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    spin_texture_magnitude = dos_non_colinear.compute_spin_texture_magnitude(atoms=atoms, orbitals=orbitals, norm_mode="spin_magnitude", from_total=True, **kwargs)
    
    array = spin_texture_magnitude.to_array()
    plotter = DOSPlotter(orientation="horizontal")
    
    
    plotter.plot(total, scalars_data=spin_texture_magnitude, scalars_mode="line")
    plotter.show() 


###########################################################
# SX magnitude testing
###########################################################

def test_non_colinear_plot_total_with_sx_magnitude_scalars_line():
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    sx = dos_non_colinear.compute_spin_texture(atoms=atoms, orbitals=orbitals, spins=[1])
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=sx, scalars_mode="line")
    plotter.show()

def test_non_spin_polarized():
    dos_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_spin_polarized.total
    projected_sum = dos_spin_polarized.compute_projected_sum(atoms=atoms, 
                                                             orbitals=orbitals, 
                                                             spins=[0,1], 
                                                             norm_mode="total_projection")
    
    plotter = DOSPlotter(orientation="horizontal")
    # plotter.plot_scalar_line(total.points, 
    #                          total.to_array()[:,0], 
    #                          projected_sum.to_array()[:,0],
    #                          alpha = [1.0,1.0])
    
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line", alpha = [0.5, 1.0], b = [0.1, 5.0])
    plotter.show()
    
    
    # sx = dos_non_spin_polarized.compute_spin_texture(atoms=atoms, orbitals=orbitals, spins=[1])


# test_non_spin_polarized()
###########################################################
# Basic plots testing
###########################################################
# test_plot_horizontal_total_line()
# test_plot_horizontal_projected_sum_line()
# test_plot_horizontal_projected_sum_line_integral_normalized()

# test_plot_horizontal_total_with_projected_sum_scalars_line_with_grouped_kwargs()
# test_plot_horizontal_total_with_projected_sum_scalars_line_flip_channel_mode()
# test_plot_horizontal_total_with_projected_sum_scalars_fill_flip_channel_mode()
# test_plot_horizontal_projected_sum_with_grouped_kwargs()
# test_plot_horizontal_total_with_projected_sum_scalars_fill_with_grouped_kwargs()

# test_plot_horizontal_total_with_projected_sum_scalars_line_flip_channel_mode_per_channel_colorbar()

###########################################################
# Orientation testing
###########################################################
# test_plot_horizontal_total_with_projected_sum_scalars_line()
# test_plot_horizontal_total_with_projected_sum_scalars_fill()

# test_plot_vertical_total_with_projected_sum_scalars_line()
# test_plot_vertical_total_with_projected_sum_scalars_fill()


# Gradient testing
# test_non_spin_polarized_total_with_gradients_line()


###########################################################
# Non-colinear testing
###########################################################

# # Projected sum
# test_non_colinear_plot_total_with_projected_sum_scalars_line()


# # Magnetization
# test_non_colinear_plot_total_with_magnetization_scalars_line()
# test_non_colinear_plot_total_with_mag_norm_mode_magnetization_from_total_line()
# test_non_colinear_plot_total_with_spin_texture_norm_mode_magnetization_scalars_line()  # This should produce values greater than one  since sum |m| <= total M

# # Spin texture magnitude
# test_non_colinear_plot_total_with_spin_texture_magnitude_scalars_line()
# test_non_colinear_plot_total_with_spin_mag_norm_mode_spin_texture_magnitude_scalars_line(fill_value=0.0)
# test_non_colinear_plot_total_with_spin_mag_norm_mode_spin_texture_magnitude_from_total_scalars_line(fill_value=0.0)  # Should result in 0.0 for all values as the total spin channels are 0.0

# test_non_colinear_plot_total_with_mag_norm_mode_spin_texture_magnitude_scalars_line()   # This should be less than one  since sum |m| <= total M

# test_non_colinear_plot_total_with_sx_magnitude_scalars_line()




#--------------------------------------------------------
# Gradient testing
#--------------------------------------------------------
# test_non_spin_polarized_total_with_gradients_line(
#     # scale = [1.0,1.0]
#     # plot_kwargs = [{ "scale": 1.0}, { "scale": 1.0}]
#     # scale = [0.001,0.001]
#     scale = [500,500]
#     )


