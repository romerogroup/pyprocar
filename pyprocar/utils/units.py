from scipy import constants

AU_TO_ANG = constants.physical_constants["Bohr radius"][0] / constants.angstrom  # bohr/ang
ANG_TO_AU = 1.0 / AU_TO_ANG  # ang/bohr


HARTREE_TO_EV = constants.physical_constants["Hartree energy in eV"][0]  # eV/Hartree
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV  # Hartree/eV

RYDBERG_TO_EV = constants.physical_constants["Rydberg constant times hc in eV"][0]

# Band gradients reach these constants in eV*m with angular k (k = 2*pi*f.b). reciprocal_lattice
# stores b without the 2*pi, so gradient code must restore it before dividing by hbar.
HBAR_EV = constants.hbar / constants.e  # eV*s
HBAR_J = constants.hbar  # J*s
METER_ANGSTROM = constants.angstrom  # m/Angstrom
EV_TO_J = constants.e  # J/eV
FREE_ELECTRON_MASS = constants.m_e  # kg
