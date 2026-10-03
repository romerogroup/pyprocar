import pytest

from pyprocar.io.abinit import AbinitDOS

DOS_TOTAL = """#
# ABINIT package : DOS file
#
# nsppol = 1, nkpt =  120, nband(1)=  30
# Tetrahedron method
# For identification : eigen(1:3)=  -2.866  -1.611  -1.610
#
# Fermi energy :       0.25000000
#
# The local DOS (in electrons/Hartree for one atomic sphere)
# and integrated local DOS (in electrons for one atomic sphere) are computed.
# at 3 energies (in Hartree) covering the interval
# between   -0.5000 and    0.5000 Hartree by steps of  0.50000 Hartree.
#
# energy(Ha)     DOS  integrated DOS
   -0.50000     2.0000     0.0000
    0.00000     4.0000     1.5000
    0.50000     8.0000     4.5000
"""


def test_abinit_dos_reads_hartree_file_as_absolute_ev(tmp_path):
    (tmp_path / "abinito_DOS_TOTAL").write_text(DOS_TOTAL)

    dos = AbinitDOS(tmp_path)

    assert dos.energies == pytest.approx([-13.605693123, 0.0, 13.605693123])
    assert dos.fermi == pytest.approx(6.8028465615)
    assert dos.dos_total[:, 0] == pytest.approx([0.0734986444, 0.1469972887, 0.2939945774])
