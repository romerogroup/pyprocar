import copy
import logging
import time
from pathlib import Path

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
    atoms = [1]
    orbitals = [4,5,6,7,8]
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0])
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(projected_sum)
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
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], normalize=True)
    
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()
    
def test_plot_horizontal_total_with_projected_sum_scalars_fill():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)

    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], normalize=True)
    
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="fill")
    plotter.show()
    

def test_plot_vertical_total_with_projected_sum_scalars_line():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)

    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], normalize=True)
    
    
    plotter = DOSPlotter(orientation="vertical")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()
    
def test_plot_vertical_total_with_projected_sum_scalars_fill():
    dos_non_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_SPIN_POLARIZED_DIR)

    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_non_spin_polarized.total
    projected_sum = dos_non_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], normalize=True)
    
    
    plotter = DOSPlotter(orientation="vertical")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="fill")
    plotter.show()
    
    
    

def test_spin_polarized_plot_total_with_projected_sum_scalars_line():
    dos_spin_polarized = DensityOfStates.from_code(code="vasp", dirpath=DOS_SPIN_POLARIZED_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]


    total = dos_spin_polarized.total
    projected_sum = dos_spin_polarized.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], normalize=True)
    
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()


def test_non_colinear_plot_total_with_projected_sum_scalars_line():
    dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=DOS_NON_COLINEAR_DIR)
    atoms = [1]
    orbitals = [4,5,6,7,8]
    
    total = dos_non_colinear.total
    projected_sum = dos_non_colinear.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0], normalize=True)
    
    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")
    plotter.show()



# test_plot_horizontal_total_line()
# test_plot_horizontal_projected_sum_line()
test_plot_horizontal_projected_sum_line_integral_normalized()

# test_plot_horizontal_total_with_projected_sum_scalars_line()
# test_plot_horizontal_total_with_projected_sum_scalars_fill()

# test_plot_vertical_total_with_projected_sum_scalars_line()
# test_plot_vertical_total_with_projected_sum_scalars_fill()


# test_plot_total_with_projected_sum_scalars_line_vertical()






# dos_non_colinear = DensityOfStates.from_code(code="vasp", dirpath=NON_COLINEAR_DIR)

# atoms = [1]
# orbitals = [4,5,6,7,8]

# projected_sum = dos_non_colinear.compute_projected_sum(atoms=atoms, orbitals=orbitals, spins=[0])
# total = dos_non_colinear.total
# plotter = DOSPlotter(orientation="horizontal")
# # plotter.plot(projected_sum,  label = "d")
# # plotter.plot(total)
# plotter.plot(total, scalars_data=projected_sum)
# # plotter.legend()
# plotter.show()


# n_e = dos_non_colinear.energies.shape[0]
# import matplotlib.pyplot as plt
# plt.plot(dos_non_colinear.energies, dos_non_colinear.total[:,0].reshape(n_e, -1), color="red")
# plt.plot(dos_non_colinear.energies, dos_non_colinear.projected[:,0,...].reshape(n_e, -1))
# plt.show()
# plotter = DOSPlotter(orientation="vertical")
# plotter.parametric_line(
#     energies=dos_non_colinear.energies,
#     dos_values=dos_non_colinear.total,
#     scalars=dos_parametric,
# )
# plotter.show()

# def validate_selection(self, selection: SelectionInput) -> list[list[int]]:
#         val_select = []
#         has_list = False
#         for element in selection:
#             if isinstance(element, list):
#                 has_list = True
#         for element in selection:
#             if isinstance(element, list):
#                 val_select.append(element)
#             elif isinstance(element, int) and has_list:
#                 val_select.append([element])
#             elif isinstance(element, int) and not has_list:
#                 val_select.append(element)
#         return val_select