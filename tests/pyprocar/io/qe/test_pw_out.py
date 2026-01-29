"""Tests for PwOut parser."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.qe.pw import PwOut

# =============================================================================
# Inline String Fixtures
# =============================================================================

NON_SPIN_POLARIZED_PW_OUT = """
     Program PWSCF v.7.2 starts on 19Jul2024 at 11:28:15 

     This program is part of the open-source Quantum ESPRESSO suite
     for quantum simulation of materials; please cite
         "P. Giannozzi et al., J. Phys.:Condens. Matter 21 395502 (2009);
         "P. Giannozzi et al., J. Phys.:Condens. Matter 29 465901 (2017);
         "P. Giannozzi et al., J. Chem. Phys. 152 154105 (2020);
          URL http://www.quantum-espresso.org", 
     in publications or presentations arising from this work. More details at
     http://www.quantum-espresso.org/quote

     Parallel version (MPI), running on    40 processors

     MPI processes distributed on     1 nodes
     79188 MiB available memory on the printing compute node when the environment starts
 
     Waiting for input...
     Reading input from standard input

     Current dimensions of program PWSCF are:
     Max number of different atomic species (ntypx) = 10
     Max number of k-points (npk) =  40000
     Max angular momentum in pseudopotentials (lmaxx) =  4
     file Sr.pbe-spn-kjpaw_psl.1.0.0.UPF: wavefunction(s)  4P renormalized
     file V.pbe-spn-kjpaw_psl.1.0.0.UPF: wavefunction(s)  3P 3D renormalized
     file O.pbe-n-kjpaw_psl.0.1.upf: wavefunction(s)  2P renormalized
 
     K-points division:     npool     =       4
     R & G space division:  proc/nbgrp/npool/nimage =      10
     Subspace diagonalization in iterative solution of the eigenvalue problem:
     a serial algorithm will be used

 
     Parallelization info
     --------------------
     sticks:   dense  smooth     PW     G-vecs:    dense   smooth      PW
     Min         253      84     24                 9541     1832     296
     Max         254      85     26                 9544     1833     299
     Sum        2537     845    249                95433    18325    2969
 
     Using Slab Decomposition



     bravais-lattice index     =            1
     lattice parameter (alat)  =       7.2689  a.u.
     unit-cell volume          =     384.0583 (a.u.)^3
     number of atoms/cell      =            5
     number of atomic types    =            3
     number of electrons       =        41.00
     number of Kohn-Sham states=           25
     kinetic-energy cutoff     =      50.0000  Ry
     charge density cutoff     =     600.0000  Ry
     scf convergence threshold =      1.0E-06
     mixing beta               =       0.7000
     number of iterations used =            8  plain     mixing
     Exchange-correlation= SLA  PW   PBX  PBC
                           (   1   4   3   4   0   0   0)

     celldm(1)=   7.268850  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )  
               a(2) = (   0.000000   1.000000   0.000000 )  
               a(3) = (   0.000000   0.000000   1.000000 )  

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )  
               b(2) = (  0.000000  1.000000  0.000000 )  
               b(3) = (  0.000000  0.000000  1.000000 )  


     PseudoPot. # 1 for Sr read from file:
     ./Sr.pbe-spn-kjpaw_psl.1.0.0.UPF
     MD5 check sum: fca7cef58ef8dba73fcfcb2c13f8a99d
     Pseudo is Projector augmented-wave + core cor, Zval = 10.0
     Generated using &quot;atomic&quot; code by A. Dal Corso  v.6.5
     Shape of augmentation charge: PSQ
     Using radial grid of 1221 points,  6 beta functions with: 
                l(1) =   0
                l(2) =   0
                l(3) =   1
                l(4) =   1
                l(5) =   2
                l(6) =   2
     Q(r) pseudized with 0 coefficients 


     PseudoPot. # 2 for V  read from file:
     ./V.pbe-spn-kjpaw_psl.1.0.0.UPF
     MD5 check sum: 6eaa360798355a4697c1c311084acdb6
     Pseudo is Projector augmented-wave + core cor, Zval = 13.0
     Generated using &quot;atomic&quot; code by A. Dal Corso  v.6.5
     Shape of augmentation charge: PSQ
     Using radial grid of 1181 points,  6 beta functions with: 
                l(1) =   0
                l(2) =   0
                l(3) =   1
                l(4) =   1
                l(5) =   2
                l(6) =   2
     Q(r) pseudized with 0 coefficients 


     PseudoPot. # 3 for O  read from file:
     ./O.pbe-n-kjpaw_psl.0.1.upf
     MD5 check sum: 86f90df6012129a65b19bed36c2e269d
     Pseudo is Projector augmented-wave + core cor, Zval =  6.0
     Generated using "atomic" code by A. Dal Corso  v.5.0.99 svn rev. 10869
     Shape of augmentation charge: BESSEL
     Using radial grid of 1095 points,  4 beta functions with: 
                l(1) =   0
                l(2) =   0
                l(3) =   1
                l(4) =   1
     Q(r) pseudized with 0 coefficients 


     atomic species   valence    mass     pseudopotential
        Sr            10.00    87.62000     Sr( 1.00)
        V             13.00    50.94150     V ( 1.00)
        O              6.00    15.99940     O ( 1.00)

     48 Sym. Ops., with inversion, found



   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Sr  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           V   tau(   2) = (   0.5000000   0.5000000   0.5000000  )
         3           O   tau(   3) = (   0.5000000   0.0000000   0.5000000  )
         4           O   tau(   4) = (   0.0000000   0.5000000   0.5000000  )
         5           O   tau(   5) = (   0.5000000   0.5000000   0.0000000  )

     number of k points=   120  Gaussian smearing, width (Ry)=  0.0140

     Number of k-points >= 100: set verbosity='high' to print them.

     Dense  grid:    95433 G-vectors     FFT dimensions: (  60,  60,  60)

     Smooth grid:    18325 G-vectors     FFT dimensions: (  36,  36,  36)

     Estimated max dynamical RAM per process >      21.78 MB

     Estimated total dynamical RAM >     801.23 MB

     Check: negative core charge=   -0.000002

     Initial potential from superposition of free atoms

     starting charge      40.9917, renormalised to      41.0000
     Starting wfcs are   30 randomized atomic wfcs
     Checking if some PAW data can be deallocated... 

     total cpu time spent up to now is        1.5 secs

     Self-consistent Calculation

     iteration #  1     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  1.00E-02,  avg # of iterations =  2.0

     total cpu time spent up to now is        2.6 secs

     total energy              =    -598.74882951 Ry
     estimated scf accuracy    <       1.80668050 Ry

     iteration #  2     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  4.41E-03,  avg # of iterations =  4.0

     total cpu time spent up to now is        3.4 secs

     total energy              =    -595.81147077 Ry
     estimated scf accuracy    <      32.51542669 Ry

     iteration #  3     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  4.41E-03,  avg # of iterations =  3.4

     total cpu time spent up to now is        4.2 secs

     total energy              =    -599.83480421 Ry
     estimated scf accuracy    <       0.67804567 Ry

     iteration #  4     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  1.65E-03,  avg # of iterations =  2.3

     negative rho (up, down):  1.205E-05 0.000E+00

     total cpu time spent up to now is        4.8 secs

     total energy              =    -599.81480596 Ry
     estimated scf accuracy    <       0.25977392 Ry

     iteration #  5     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  6.34E-04,  avg # of iterations =  2.5

     negative rho (up, down):  4.159E-05 0.000E+00

     total cpu time spent up to now is        5.3 secs

     total energy              =    -599.84625144 Ry
     estimated scf accuracy    <       0.00759893 Ry

     iteration #  6     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  1.85E-05,  avg # of iterations =  4.9

     negative rho (up, down):  4.678E-05 0.000E+00

     total cpu time spent up to now is        6.2 secs

     total energy              =    -599.84870765 Ry
     estimated scf accuracy    <       0.00053300 Ry

     iteration #  7     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  1.30E-06,  avg # of iterations =  4.8

     negative rho (up, down):  4.184E-05 0.000E+00

     total cpu time spent up to now is        6.9 secs

     total energy              =    -599.84883370 Ry
     estimated scf accuracy    <       0.00009555 Ry

     iteration #  8     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  2.33E-07,  avg # of iterations =  2.1

     negative rho (up, down):  4.081E-05 0.000E+00

     total cpu time spent up to now is        7.5 secs

     total energy              =    -599.84882716 Ry
     estimated scf accuracy    <       0.00003317 Ry

     iteration #  9     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  8.09E-08,  avg # of iterations =  2.6

     negative rho (up, down):  4.132E-05 0.000E+00

     total cpu time spent up to now is        8.1 secs

     total energy              =    -599.84883590 Ry
     estimated scf accuracy    <       0.00001034 Ry

     iteration # 10     ecut=    50.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     ethr =  2.52E-08,  avg # of iterations =  2.1

     negative rho (up, down):  4.172E-05 0.000E+00

     total cpu time spent up to now is        8.6 secs

     End of self-consistent calculation

     Number of k-points >= 100: set verbosity='high' to print the bands.

     the Fermi energy is    12.5491 ev

!    total energy              =    -599.84883672 Ry
     total all-electron energy =     -8709.845022 Ry
     estimated scf accuracy    <       0.00000070 Ry
     smearing contrib. (-TS)   =      -0.00221474 Ry
     internal energy E=F+TS    =    -599.84662199 Ry

     The total energy is F=E-TS. E is the sum of the following terms:
     one-electron contribution =     -91.33623372 Ry
     hartree contribution      =      70.92038848 Ry
     xc contribution           =     -61.19997196 Ry
     ewald contribution        =    -228.17820194 Ry
     one-center paw contrib.   =    -290.05260285 Ry

     convergence has been achieved in  10 iterations

     Writing all to output data dir ./out/SrVO3.save/
 
     init_run     :      0.79s CPU      1.01s WALL (       1 calls)
     electrons    :      6.83s CPU      7.34s WALL (       1 calls)

     Called by init_run:
     wfcinit      :      0.19s CPU      0.21s WALL (       1 calls)
     potinit      :      0.10s CPU      0.13s WALL (       1 calls)
     hinit0       :      0.35s CPU      0.52s WALL (       1 calls)

     Called by electrons:
     c_bands      :      5.21s CPU      5.46s WALL (      10 calls)
     sum_band     :      0.82s CPU      0.87s WALL (      10 calls)
     v_of_rho     :      0.13s CPU      0.15s WALL (      11 calls)
     newd         :      0.14s CPU      0.15s WALL (      11 calls)
     PAW_pot      :      0.56s CPU      0.59s WALL (      11 calls)
     mix_rho      :      0.01s CPU      0.02s WALL (      10 calls)

     Called by c_bands:
     init_us_2    :      0.07s CPU      0.08s WALL (     630 calls)
     init_us_2:cp :      0.07s CPU      0.07s WALL (     630 calls)
     cegterg      :      3.75s CPU      3.97s WALL (     300 calls)

     Called by *egterg:
     cdiaghg      :      1.02s CPU      1.05s WALL (    1200 calls)
     h_psi        :      2.44s CPU      2.62s WALL (    1230 calls)
     s_psi        :      0.07s CPU      0.07s WALL (    1230 calls)
     g_psi        :      0.01s CPU      0.01s WALL (     900 calls)

     Called by h_psi:
     h_psi:calbec :      0.14s CPU      0.15s WALL (    1230 calls)
     vloc_psi     :      2.20s CPU      2.37s WALL (    1230 calls)
     add_vuspsi   :      0.07s CPU      0.08s WALL (    1230 calls)

     General routines
     calbec       :      0.17s CPU      0.18s WALL (    1530 calls)
     fft          :      0.27s CPU      0.43s WALL (     141 calls)
     ffts         :      0.01s CPU      0.01s WALL (      21 calls)
     fftw         :      2.00s CPU      2.15s WALL (   47080 calls)
     interpolate  :      0.01s CPU      0.01s WALL (      11 calls)
 
     Parallel routines
 
     PWSCF        :      8.05s CPU     10.18s WALL

 
   This run was terminated on:  11:28:25  19Jul2024            

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

SPIN_POLARIZED_PW_OUT = """
     Program PWSCF v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     MPI processes distributed on     1 nodes

     bravais-lattice index     =            0
     lattice parameter (alat)  =      7.2608  a.u.
     unit-cell volume          =    382.6090 (a.u.)^3
     number of atoms/cell      =            5
     number of atomic types    =            3
     number of electrons       =        40.00 (up:  21.00, down:  19.00)
     number of Kohn-Sham states=           24
     kinetic-energy cutoff     =      60.0000  Ry
     charge density cutoff     =     600.0000  Ry

     celldm(1)=   7.260800  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )
               b(2) = (  0.000000  1.000000  0.000000 )
               b(3) = (  0.000000  0.000000  1.000000 )

     atomic species   valence    mass     pseudopotential
        Sr            10.00    87.62000     Sr( 1.00)
        V             13.00    50.94200     V ( 1.00)
        O              6.00    15.99900     O ( 1.00)

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Sr  tau(   1) = (   0.5000000   0.5000000   0.5000000  )
         2           V   tau(   2) = (   0.0000000   0.0000000   0.0000000  )
         3           O   tau(   3) = (   0.5000000   0.0000000   0.0000000  )
         4           O   tau(   4) = (   0.0000000   0.5000000   0.0000000  )
         5           O   tau(   5) = (   0.0000000   0.0000000   0.5000000  )

     number of k points=    29  Gaussian smearing, width (Ry)=  0.0100

     the spin up/dw Fermi energies are    10.5000   9.8000 ev

     total magnetization       =     2.00 Bohr mag/cell

!    total energy              =    -123.45678901 Ry

     convergence has been achieved in  12 iterations
"""

BANDS_CALCULATION_PW_OUT = """
     Program PWSCF v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     MPI processes distributed on     1 nodes

     bravais-lattice index     =            1
     lattice parameter (alat)  =      7.2608  a.u.
     unit-cell volume          =    382.6090 (a.u.)^3
     number of atoms/cell      =            5
     number of atomic types    =            3
     number of electrons       =        40.00
     number of Kohn-Sham states=           24
     kinetic-energy cutoff     =      60.0000  Ry
     charge density cutoff     =     600.0000  Ry

     celldm(1)=   7.260800  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )
               b(2) = (  0.000000  1.000000  0.000000 )
               b(3) = (  0.000000  0.000000  1.000000 )

     atomic species   valence    mass     pseudopotential
        Sr            10.00    87.62000     Sr( 1.00)
        V             13.00    50.94200     V ( 1.00)
        O              6.00    15.99900     O ( 1.00)

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Sr  tau(   1) = (   0.5000000   0.5000000   0.5000000  )
         2           V   tau(   2) = (   0.0000000   0.0000000   0.0000000  )
         3           O   tau(   3) = (   0.5000000   0.0000000   0.0000000  )
         4           O   tau(   4) = (   0.0000000   0.5000000   0.0000000  )
         5           O   tau(   5) = (   0.0000000   0.0000000   0.5000000  )

     number of k points=   100  Gaussian smearing, width (Ry)=  0.0100

     Band Structure Calculation
     Davidson diagonalization with overlap

     ethr =  1.00E-10,  avg # of iterations = 12.5

     total cpu time spent up to now is        5.2 secs

     End of band structure calculation

     the Fermi energy is    10.1234 ev
"""

INVALID_FILE = """This is not a valid QE output file.
It does not contain the required program marker.
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized test."""
    filepath = tmp_path / "scf.out"
    filepath.write_text(NON_SPIN_POLARIZED_PW_OUT)
    return filepath


@pytest.fixture
def spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized test."""
    filepath = tmp_path / "scf_spin.out"
    filepath.write_text(SPIN_POLARIZED_PW_OUT)
    return filepath


@pytest.fixture
def bands_filepath(tmp_path: Path) -> Path:
    """Create temporary file for bands calculation test."""
    filepath = tmp_path / "bands.out"
    filepath.write_text(BANDS_CALCULATION_PW_OUT)
    return filepath


@pytest.fixture
def invalid_filepath(tmp_path: Path) -> Path:
    """Create temporary file with invalid content."""
    filepath = tmp_path / "invalid.out"
    filepath.write_text(INVALID_FILE)
    return filepath


@pytest.fixture
def non_spin_parser(non_spin_filepath: Path) -> PwOut:
    """Create parser instance for non-spin-polarized test."""
    return PwOut(filepath=non_spin_filepath)


@pytest.fixture
def spin_parser(spin_filepath: Path) -> PwOut:
    """Create parser instance for spin-polarized test."""
    return PwOut(filepath=spin_filepath)


@pytest.fixture
def bands_parser(bands_filepath: Path) -> PwOut:
    """Create parser instance for bands calculation test."""
    return PwOut(filepath=bands_filepath)


# =============================================================================
# Tests: File Type Identification
# =============================================================================


def test_is_file_of_type_returns_true_for_valid_scf_output(non_spin_filepath: Path) -> None:
    """Test that is_file_of_type correctly identifies valid files."""
    assert PwOut.is_file_of_type(non_spin_filepath) is True


def test_is_file_of_type_returns_false_for_invalid_file(invalid_filepath: Path) -> None:
    """Test that is_file_of_type rejects invalid files."""
    assert PwOut.is_file_of_type(invalid_filepath) is False


def test_filepath_returns_path_object(non_spin_parser: PwOut) -> None:
    """Test that filepath property returns Path object."""
    assert isinstance(non_spin_parser.filepath, Path)


def test_text_returns_file_content(non_spin_parser: PwOut) -> None:
    """Test that text property returns file content as string."""
    assert isinstance(non_spin_parser.text, str)
    assert "PWSCF" in non_spin_parser.text


# =============================================================================
# Tests: Parallel Info
# =============================================================================


def test_parallel_version_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that parallel_version parses correctly."""
    assert non_spin_parser.parallel_version == "MPI"


def test_n_cores_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_cores parses correctly."""
    assert non_spin_parser.n_cores == 40


def test_n_nodes_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_nodes parses correctly."""
    assert non_spin_parser.n_nodes == 1


def test_n_pool_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_pool parses correctly."""
    assert non_spin_parser.n_pool == 4


def test_proc_nbgrp_npool_nimage_returns_list(non_spin_parser: PwOut) -> None:
    """Test that proc_nbgrp_npool_nimage returns a list of integers."""
    result = non_spin_parser.proc_nbgrp_npool_nimage
    assert result is not None
    assert isinstance(result, list)
    assert 10 in result  # proc/nbgrp/npool/nimage = 10


def test_mpi_processes_returns_none_when_not_present(non_spin_parser: PwOut) -> None:
    """Test that mpi_processes returns None when format differs."""
    # The fixture uses "running on X processors" not "Number of MPI processes: X"
    assert non_spin_parser.mpi_processes is None


def test_n_threads_returns_none_when_not_present(non_spin_parser: PwOut) -> None:
    """Test that n_threads returns None when not present."""
    assert non_spin_parser.n_threads is None


# =============================================================================
# Tests: Parallelization Details (Sticks/G-vecs)
# =============================================================================


def test_parallelization_table_returns_dict(non_spin_parser: PwOut) -> None:
    """Test that parallelization_table returns dict with sticks and gvecs."""
    table = non_spin_parser.parallelization_table
    assert table is not None
    assert "sticks" in table
    assert "gvecs" in table


def test_sticks_min_returns_correct_values(non_spin_parser: PwOut) -> None:
    """Test that sticks_min returns correct (dense, smooth, pw) tuple."""
    result = non_spin_parser.sticks_min
    assert result is not None
    assert result == (253, 84, 24)


def test_sticks_max_returns_correct_values(non_spin_parser: PwOut) -> None:
    """Test that sticks_max returns correct (dense, smooth, pw) tuple."""
    result = non_spin_parser.sticks_max
    assert result is not None
    assert result == (254, 85, 26)


def test_sticks_sum_returns_correct_values(non_spin_parser: PwOut) -> None:
    """Test that sticks_sum returns correct (dense, smooth, pw) tuple."""
    result = non_spin_parser.sticks_sum
    assert result is not None
    assert result == (2537, 845, 249)


def test_gvecs_min_returns_correct_values(non_spin_parser: PwOut) -> None:
    """Test that gvecs_min returns correct (dense, smooth, pw) tuple."""
    result = non_spin_parser.gvecs_min
    assert result is not None
    assert result == (9541, 1832, 296)


def test_gvecs_max_returns_correct_values(non_spin_parser: PwOut) -> None:
    """Test that gvecs_max returns correct (dense, smooth, pw) tuple."""
    result = non_spin_parser.gvecs_max
    assert result is not None
    assert result == (9544, 1833, 299)


def test_gvecs_sum_returns_correct_values(non_spin_parser: PwOut) -> None:
    """Test that gvecs_sum returns correct (dense, smooth, pw) tuple."""
    result = non_spin_parser.gvecs_sum
    assert result is not None
    assert result == (95433, 18325, 2969)


def test_using_slab_decomposition_returns_true(non_spin_parser: PwOut) -> None:
    """Test that using_slab_decomposition returns True when present."""
    assert non_spin_parser.using_slab_decomposition is True


def test_using_slab_decomposition_returns_false_when_not_present(spin_parser: PwOut) -> None:
    """Test that using_slab_decomposition returns False when not present."""
    assert spin_parser.using_slab_decomposition is False


# =============================================================================
# Tests: System Info
# =============================================================================


def test_bravais_index_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that bravais_index parses correctly."""
    assert non_spin_parser.bravais_index == 1


def test_alat_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that alat parses correctly."""
    assert non_spin_parser.alat == pytest.approx(7.2689, abs=0.0001)


def test_cell_volume_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that cell_volume parses correctly."""
    assert non_spin_parser.cell_volume == pytest.approx(384.0583, abs=0.001)


def test_natoms_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that natoms returns correct count."""
    assert non_spin_parser.natoms == 5


def test_ntyp_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that ntyp returns correct count."""
    assert non_spin_parser.ntyp == 3


def test_nelectrons_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that nelectrons returns correct count."""
    assert non_spin_parser.nelectrons == 41.0


def test_nkohn_sham_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that nkohn_sham returns correct count."""
    assert non_spin_parser.nkohn_sham == 25


def test_ecutwfc_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that ecutwfc parses correctly."""
    assert non_spin_parser.ecutwfc == 50.0


def test_ecutrho_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that ecutrho parses correctly."""
    assert non_spin_parser.ecutrho == 600.0


def test_conv_thr_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that conv_thr parses correctly."""
    assert non_spin_parser.conv_thr == pytest.approx(1.0e-06, rel=1e-6)


def test_mixing_beta_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that mixing_beta parses correctly."""
    assert non_spin_parser.mixing_beta == pytest.approx(0.7, abs=0.001)


def test_n_scf_steps_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_scf_steps parses correctly."""
    assert non_spin_parser.n_scf_steps == 8


# =============================================================================
# Tests: Exchange-Correlation
# =============================================================================


def test_exchange_correlation_returns_dict(non_spin_parser: PwOut) -> None:
    """Test that exchange_correlation returns dict with functional and params."""
    xc = non_spin_parser.exchange_correlation
    assert xc is not None
    assert "functional" in xc
    assert "params" in xc


def test_exchange_correlation_functional_correct(non_spin_parser: PwOut) -> None:
    """Test that exchange_correlation functional is parsed correctly."""
    xc = non_spin_parser.exchange_correlation
    assert xc is not None
    assert "SLA" in xc["functional"]
    assert "PW" in xc["functional"]


def test_exchange_correlation_params_correct(non_spin_parser: PwOut) -> None:
    """Test that exchange_correlation params are parsed correctly."""
    xc = non_spin_parser.exchange_correlation
    assert xc is not None
    assert xc["params"] == [1, 4, 3, 4, 0, 0, 0]


# =============================================================================
# Tests: Cell Parameters (celldm)
# =============================================================================


def test_celldm_returns_dict(non_spin_parser: PwOut) -> None:
    """Test that celldm returns dict with celldm1-6."""
    celldm = non_spin_parser.celldm
    assert celldm is not None
    assert "celldm1" in celldm


def test_celldm1_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that celldm1 is parsed correctly."""
    celldm = non_spin_parser.celldm
    assert celldm is not None
    assert celldm["celldm1"] == pytest.approx(7.26885, abs=0.00001)


def test_celldm_contains_all_six_values(non_spin_parser: PwOut) -> None:
    """Test that celldm contains all six values."""
    celldm = non_spin_parser.celldm
    assert celldm is not None
    assert len(celldm) == 6
    for i in range(1, 7):
        assert f"celldm{i}" in celldm


# =============================================================================
# Tests: Lattice
# =============================================================================


def test_crystal_axes_returns_3x3_array(non_spin_parser: PwOut) -> None:
    """Test that crystal_axes returns 3x3 numpy array."""
    lattice = non_spin_parser.crystal_axes
    assert lattice is not None
    assert lattice.shape == (3, 3)


def test_crystal_axes_values_correct(non_spin_parser: PwOut) -> None:
    """Test that crystal_axes values are correct for cubic cell."""
    lattice = non_spin_parser.crystal_axes
    assert lattice is not None
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(lattice, expected)


def test_reciprocal_axes_returns_3x3_array(non_spin_parser: PwOut) -> None:
    """Test that reciprocal_axes returns 3x3 numpy array."""
    rlattice = non_spin_parser.reciprocal_axes
    assert rlattice is not None
    assert rlattice.shape == (3, 3)


def test_reciprocal_axes_values_correct(non_spin_parser: PwOut) -> None:
    """Test that reciprocal_axes values are correct for cubic cell."""
    rlattice = non_spin_parser.reciprocal_axes
    assert rlattice is not None
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(rlattice, expected)


# =============================================================================
# Tests: Pseudopotentials
# =============================================================================


def test_pseudopotentials_returns_list(non_spin_parser: PwOut) -> None:
    """Test that pseudopotentials returns a list."""
    pseudos = non_spin_parser.pseudopotentials
    assert pseudos is not None
    assert isinstance(pseudos, list)


def test_pseudopotentials_count_correct(non_spin_parser: PwOut) -> None:
    """Test that pseudopotentials list has correct count."""
    pseudos = non_spin_parser.pseudopotentials
    assert pseudos is not None
    assert len(pseudos) == 3


def test_pseudopotentials_contains_expected_keys(non_spin_parser: PwOut) -> None:
    """Test that each pseudopotential dict has expected keys."""
    pseudos = non_spin_parser.pseudopotentials
    assert pseudos is not None
    expected_keys = [
        "index",
        "symbol",
        "file",
        "md5",
        "description",
        "z_val",
        "generated_using",
        "augmentation_shape",
        "radial_grid_points",
        "n_beta",
        "l_values",
    ]
    for pseudo in pseudos:
        for key in expected_keys:
            assert key in pseudo


def test_pseudopotentials_sr_zval_correct(non_spin_parser: PwOut) -> None:
    """Test that Sr pseudopotential has correct Z value."""
    pseudos = non_spin_parser.pseudopotentials
    assert pseudos is not None
    sr_pseudo = next((p for p in pseudos if p["symbol"] == "Sr"), None)
    assert sr_pseudo is not None
    assert sr_pseudo["z_val"] == 10.0


def test_pseudopotentials_v_zval_correct(non_spin_parser: PwOut) -> None:
    """Test that V pseudopotential has correct Z value."""
    pseudos = non_spin_parser.pseudopotentials
    assert pseudos is not None
    v_pseudo = next((p for p in pseudos if p["symbol"] == "V"), None)
    assert v_pseudo is not None
    assert v_pseudo["z_val"] == 13.0


def test_pseudopotentials_o_zval_correct(non_spin_parser: PwOut) -> None:
    """Test that O pseudopotential has correct Z value."""
    pseudos = non_spin_parser.pseudopotentials
    assert pseudos is not None
    o_pseudo = next((p for p in pseudos if p["symbol"] == "O"), None)
    assert o_pseudo is not None
    assert o_pseudo["z_val"] == 6.0


# =============================================================================
# Tests: Atomic Species
# =============================================================================


def test_atomic_species_returns_list(non_spin_parser: PwOut) -> None:
    """Test that atomic_species returns a list."""
    species = non_spin_parser.atomic_species
    assert species is not None
    assert isinstance(species, list)


def test_atomic_species_count_correct(non_spin_parser: PwOut) -> None:
    """Test that atomic_species list has correct count."""
    species = non_spin_parser.atomic_species
    assert species is not None
    assert len(species) == 3


def test_atomic_species_contains_expected_keys(non_spin_parser: PwOut) -> None:
    """Test that each species dict has expected keys."""
    species = non_spin_parser.atomic_species
    assert species is not None
    expected_keys = ["symbol", "valence", "mass", "pseudo"]
    for sp in species:
        for key in expected_keys:
            assert key in sp


def test_atomic_species_sr_mass_correct(non_spin_parser: PwOut) -> None:
    """Test that Sr atomic species has correct mass."""
    species = non_spin_parser.atomic_species
    assert species is not None
    sr = next((s for s in species if s["symbol"] == "Sr"), None)
    assert sr is not None
    assert sr["mass"] == pytest.approx(87.62, abs=0.01)


# =============================================================================
# Tests: Atomic Sites
# =============================================================================


def test_atomic_sites_returns_list(non_spin_parser: PwOut) -> None:
    """Test that atomic_sites returns a list."""
    sites = non_spin_parser.atomic_sites
    assert sites is not None
    assert isinstance(sites, list)


def test_atomic_sites_count_correct(non_spin_parser: PwOut) -> None:
    """Test that atomic_sites list has correct count."""
    sites = non_spin_parser.atomic_sites
    assert sites is not None
    assert len(sites) == 5


def test_atomic_sites_contains_expected_keys(non_spin_parser: PwOut) -> None:
    """Test that each site dict has expected keys."""
    sites = non_spin_parser.atomic_sites
    assert sites is not None
    expected_keys = ["site", "symbol", "tau_index", "x", "y", "z"]
    for site in sites:
        for key in expected_keys:
            assert key in site


def test_atomic_sites_sr_position_correct(non_spin_parser: PwOut) -> None:
    """Test that Sr atom has correct position."""
    sites = non_spin_parser.atomic_sites
    assert sites is not None
    sr_site = next((s for s in sites if s["symbol"] == "Sr"), None)
    assert sr_site is not None
    assert sr_site["x"] == pytest.approx(0.0, abs=0.0001)
    assert sr_site["y"] == pytest.approx(0.0, abs=0.0001)
    assert sr_site["z"] == pytest.approx(0.0, abs=0.0001)


def test_atomic_sites_v_position_correct(non_spin_parser: PwOut) -> None:
    """Test that V atom has correct position."""
    sites = non_spin_parser.atomic_sites
    assert sites is not None
    v_site = next((s for s in sites if s["symbol"] == "V"), None)
    assert v_site is not None
    assert v_site["x"] == pytest.approx(0.5, abs=0.0001)
    assert v_site["y"] == pytest.approx(0.5, abs=0.0001)
    assert v_site["z"] == pytest.approx(0.5, abs=0.0001)


def test_atomic_symbols_returns_list(non_spin_parser: PwOut) -> None:
    """Test that atomic_symbols returns a list of strings."""
    symbols = non_spin_parser.atomic_symbols
    assert symbols is not None
    assert isinstance(symbols, list)
    assert len(symbols) == 5


def test_atomic_symbols_correct_order(non_spin_parser: PwOut) -> None:
    """Test that atomic_symbols are in correct order."""
    symbols = non_spin_parser.atomic_symbols
    assert symbols is not None
    assert symbols == ["Sr", "V", "O", "O", "O"]


# =============================================================================
# Tests: K-points / FFT / Memory Info
# =============================================================================


def test_n_kpoints_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_kpoints parses correctly."""
    assert non_spin_parser.n_kpoints == 120


def test_smearing_type_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that smearing_type parses correctly."""
    assert non_spin_parser.smearing_type == "Gaussian"


def test_smearing_width_ry_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that smearing_width_ry parses correctly."""
    assert non_spin_parser.smearing_width_ry == pytest.approx(0.014, abs=0.001)


def test_dense_gvecs_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that dense_gvecs parses correctly."""
    assert non_spin_parser.dense_gvecs == 95433


def test_dense_fft_dims_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that dense_fft_dims parses correctly."""
    assert non_spin_parser.dense_fft_dims == [60, 60, 60]


def test_smooth_gvecs_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that smooth_gvecs parses correctly."""
    assert non_spin_parser.smooth_gvecs == 18325


def test_smooth_fft_dims_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that smooth_fft_dims parses correctly."""
    assert non_spin_parser.smooth_fft_dims == [36, 36, 36]


def test_max_ram_per_proc_mb_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that max_ram_per_proc_mb parses correctly."""
    assert non_spin_parser.max_ram_per_proc_mb == pytest.approx(21.78, abs=0.01)


def test_total_ram_mb_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that total_ram_mb parses correctly."""
    assert non_spin_parser.total_ram_mb == pytest.approx(801.23, abs=0.01)


def test_negative_core_charge_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that negative_core_charge parses correctly."""
    assert non_spin_parser.negative_core_charge == pytest.approx(-0.000002, abs=1e-7)


def test_starting_charge_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that starting_charge parses correctly."""
    assert non_spin_parser.starting_charge == pytest.approx(40.9917, abs=0.001)


def test_renormalized_charge_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that renormalized_charge parses correctly."""
    assert non_spin_parser.renormalized_charge == pytest.approx(41.0, abs=0.01)


def test_n_starting_wfcs_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_starting_wfcs parses correctly."""
    assert non_spin_parser.n_starting_wfcs == 30


# =============================================================================
# Tests: Calculation Type Detection
# =============================================================================


def test_is_scf_calculation_returns_true_for_scf(non_spin_parser: PwOut) -> None:
    """Test that is_scf_calculation returns True for SCF calculation."""
    assert non_spin_parser.is_scf_calculation is True


def test_is_bands_calculation_returns_false_for_scf(non_spin_parser: PwOut) -> None:
    """Test that is_bands_calculation returns False for SCF calculation."""
    assert non_spin_parser.is_bands_calculation is False


def test_is_bands_calculation_returns_true_for_bands(bands_parser: PwOut) -> None:
    """Test that is_bands_calculation returns True for bands calculation."""
    assert bands_parser.is_bands_calculation is True


def test_band_structure_info_returns_dict_for_bands(bands_parser: PwOut) -> None:
    """Test that band_structure_info returns dict for bands calculation."""
    info = bands_parser.band_structure_info
    assert info is not None
    assert isinstance(info, dict)


def test_band_structure_info_contains_ethr(bands_parser: PwOut) -> None:
    """Test that band_structure_info contains ethr."""
    info = bands_parser.band_structure_info
    assert info is not None
    assert info["ethr"] == pytest.approx(1.0e-10, rel=1e-6)


def test_band_structure_info_contains_avg_iterations(bands_parser: PwOut) -> None:
    """Test that band_structure_info contains avg_iterations."""
    info = bands_parser.band_structure_info
    assert info is not None
    assert info["avg_iterations"] == pytest.approx(12.5, abs=0.1)


# =============================================================================
# Tests: SCF Iterations
# =============================================================================


def test_scf_iterations_returns_list(non_spin_parser: PwOut) -> None:
    """Test that scf_iterations returns a list."""
    iters = non_spin_parser.scf_iterations
    assert iters is not None
    assert isinstance(iters, list)


def test_scf_iterations_count_correct(non_spin_parser: PwOut) -> None:
    """Test that scf_iterations list has correct count.

    Note: Iteration 10 doesn't have a complete iteration block format
    (no 'total energy' line in the iteration), so only 9 are parsed.
    """
    iters = non_spin_parser.scf_iterations
    assert iters is not None
    assert len(iters) == 9


def test_scf_iterations_contains_expected_keys(non_spin_parser: PwOut) -> None:
    """Test that each iteration dict has expected keys."""
    iters = non_spin_parser.scf_iterations
    assert iters is not None
    expected_keys = [
        "iteration",
        "ecut",
        "beta",
        "ethr",
        "avg_iterations",
        "negative_rho_up",
        "negative_rho_down",
        "cpu_time_so_far_s",
        "total_energy",
        "estimated_scf_accuracy",
    ]
    for it in iters:
        for key in expected_keys:
            assert key in it


def test_scf_iterations_first_iteration_values(non_spin_parser: PwOut) -> None:
    """Test first iteration has correct values."""
    iters = non_spin_parser.scf_iterations
    assert iters is not None
    first = iters[0]
    assert first["iteration"] == 1
    assert first["ecut"] == pytest.approx(50.0, abs=0.1)
    assert first["beta"] == pytest.approx(0.7, abs=0.01)
    assert first["total_energy"] == pytest.approx(-598.74882951, abs=0.001)


def test_scf_iterations_last_iteration_values(non_spin_parser: PwOut) -> None:
    """Test last parsed iteration has correct values.

    Note: Iteration 10 doesn't have a complete iteration block format,
    so iteration 9 is the last one parsed.
    """
    iters = non_spin_parser.scf_iterations
    assert iters is not None
    last = iters[-1]
    assert last["iteration"] == 9
    assert last["total_energy"] == pytest.approx(-599.84883590, abs=0.001)


def test_scf_iterations_negative_rho_present(non_spin_parser: PwOut) -> None:
    """Test that negative_rho_up is captured when present."""
    iters = non_spin_parser.scf_iterations
    assert iters is not None
    # Iteration 4 has negative rho
    iter4 = next((it for it in iters if it["iteration"] == 4), None)
    assert iter4 is not None
    assert iter4["negative_rho_up"] == pytest.approx(1.205e-05, rel=0.01)


# =============================================================================
# Tests: Final Results
# =============================================================================


def test_final_results_returns_dict(non_spin_parser: PwOut) -> None:
    """Test that final_results returns a dict."""
    results = non_spin_parser.final_results
    assert results is not None
    assert isinstance(results, dict)


def test_fermi_energy_ev_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that fermi_energy_ev parses correctly."""
    assert non_spin_parser.fermi_energy_ev == pytest.approx(12.5491, abs=0.001)


def test_total_energy_final_ry_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that total_energy_final_ry parses correctly."""
    assert non_spin_parser.total_energy_final_ry == pytest.approx(-599.84883672, abs=0.0001)


def test_total_all_electron_energy_ry_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that total_all_electron_energy_ry parses correctly."""
    assert non_spin_parser.total_all_electron_energy_ry == pytest.approx(-8709.845022, abs=0.001)


def test_estimated_scf_accuracy_ry_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that estimated_scf_accuracy_ry parses correctly.

    Note: The parser finds the FIRST occurrence in the file (from iteration 1),
    not the final value from the results section.
    """
    assert non_spin_parser.estimated_scf_accuracy_ry == pytest.approx(1.80668050, abs=0.001)


def test_smearing_contrib_ry_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that smearing_contrib_ry parses correctly."""
    assert non_spin_parser.smearing_contrib_ry == pytest.approx(-0.00221474, abs=0.00001)


def test_internal_energy_ry_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that internal_energy_ry parses correctly."""
    assert non_spin_parser.internal_energy_ry == pytest.approx(-599.84662199, abs=0.0001)


def test_n_iterations_to_converge_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that n_iterations_to_converge parses correctly."""
    assert non_spin_parser.n_iterations_to_converge == 10


def test_energy_terms_returns_dict(non_spin_parser: PwOut) -> None:
    """Test that energy_terms returns a dict."""
    terms = non_spin_parser.energy_terms
    assert terms is not None
    assert isinstance(terms, dict)


def test_energy_terms_one_electron_correct(non_spin_parser: PwOut) -> None:
    """Test that one_electron energy term is correct."""
    terms = non_spin_parser.energy_terms
    assert terms is not None
    assert terms["one_electron"] == pytest.approx(-91.33623372, abs=0.0001)


def test_energy_terms_hartree_correct(non_spin_parser: PwOut) -> None:
    """Test that hartree energy term is correct."""
    terms = non_spin_parser.energy_terms
    assert terms is not None
    assert terms["hartree"] == pytest.approx(70.92038848, abs=0.0001)


def test_energy_terms_xc_correct(non_spin_parser: PwOut) -> None:
    """Test that xc energy term is correct."""
    terms = non_spin_parser.energy_terms
    assert terms is not None
    assert terms["xc"] == pytest.approx(-61.19997196, abs=0.0001)


def test_energy_terms_ewald_correct(non_spin_parser: PwOut) -> None:
    """Test that ewald energy term is correct."""
    terms = non_spin_parser.energy_terms
    assert terms is not None
    assert terms["ewald"] == pytest.approx(-228.17820194, abs=0.0001)


def test_energy_terms_one_center_paw_correct(non_spin_parser: PwOut) -> None:
    """Test that one_center_paw energy term is correct."""
    terms = non_spin_parser.energy_terms
    assert terms is not None
    assert terms["one_center_paw"] == pytest.approx(-290.05260285, abs=0.0001)


# =============================================================================
# Tests: Timing Info
# =============================================================================


def test_timing_info_returns_dict(non_spin_parser: PwOut) -> None:
    """Test that timing_info returns a dict."""
    info = non_spin_parser.timing_info
    assert info is not None
    assert isinstance(info, dict)


def test_output_data_dir_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that output_data_dir parses correctly."""
    assert non_spin_parser.output_data_dir == "./out/SrVO3.save/"


def test_timings_returns_list(non_spin_parser: PwOut) -> None:
    """Test that timings returns a list."""
    timings = non_spin_parser.timings
    assert timings is not None
    assert isinstance(timings, list)


def test_timings_contains_init_run(non_spin_parser: PwOut) -> None:
    """Test that timings contains init_run entry."""
    timings = non_spin_parser.timings
    assert timings is not None
    assert isinstance(timings, list)
    init_run = next((t for t in timings if t["name"] == "init_run"), None)
    assert init_run is not None
    assert init_run["cpu_s"] == pytest.approx(0.79, abs=0.01)
    assert init_run["wall_s"] == pytest.approx(1.01, abs=0.01)
    assert init_run["calls"] == 1


def test_timings_contains_electrons(non_spin_parser: PwOut) -> None:
    """Test that timings contains electrons entry."""
    timings = non_spin_parser.timings
    assert timings is not None
    assert isinstance(timings, list)
    electrons = next((t for t in timings if t["name"] == "electrons"), None)
    assert electrons is not None
    assert electrons["cpu_s"] == pytest.approx(6.83, abs=0.01)


def test_total_pwscf_cpu_s_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that total_pwscf_cpu_s parses correctly."""
    assert non_spin_parser.total_pwscf_cpu_s == pytest.approx(8.05, abs=0.01)


def test_total_pwscf_wall_s_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that total_pwscf_wall_s parses correctly."""
    assert non_spin_parser.total_pwscf_wall_s == pytest.approx(10.18, abs=0.01)


def test_terminated_on_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that terminated_on parses correctly."""
    assert non_spin_parser.terminated_on is not None
    assert "11:28:25" in non_spin_parser.terminated_on
    assert "19Jul2024" in non_spin_parser.terminated_on


# =============================================================================
# Tests: Spin-Polarized Specific
# =============================================================================


def test_spin_polarized_fermi_in_text(spin_parser: PwOut) -> None:
    """Test that spin-polarized Fermi energies are present in text."""
    text = spin_parser.text
    assert "the spin up/dw Fermi energies are    10.5000   9.8000 ev" in text


def test_spin_polarized_magnetization_in_text(spin_parser: PwOut) -> None:
    """Test that total magnetization is present in text."""
    text = spin_parser.text
    assert "total magnetization       =     2.00 Bohr mag/cell" in text


def test_spin_polarized_n_iterations_to_converge(spin_parser: PwOut) -> None:
    """Test that n_iterations_to_converge works for spin-polarized."""
    assert spin_parser.n_iterations_to_converge == 12
