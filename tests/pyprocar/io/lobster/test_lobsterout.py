"""Tests for LobsterOut extractor."""

from pyprocar.io.lobster import LobsterOut

LOBSTEROUT_CONTENT = """LOBSTER v4.1.0
detecting used PAW program... VASP
initializing projections...
calculating FatBand for Element: Fe Orbital(s): s p_y p_z p_x d_xy d_yz d_z^2 d_xz d_x^2-y^2
calculating FatBand for Element: O Orbital(s): s p_y p_z p_x
finished!
"""


class TestLobsterOut:
    def test_from_str(self):
        extractor = LobsterOut.from_str(LOBSTEROUT_CONTENT)
        assert extractor is not None

    def test_ions_list(self):
        extractor = LobsterOut.from_str(LOBSTEROUT_CONTENT)
        assert extractor.ions_list == ["Fe", "O"]

    def test_fatband_info(self):
        extractor = LobsterOut.from_str(LOBSTEROUT_CONTENT)
        info = extractor.fatband_info
        assert len(info) == 2
        assert info[0][0] == "Fe"
        assert "s" in info[0][1]
        assert info[1][0] == "O"

    def test_fatband_filenames(self):
        extractor = LobsterOut.from_str(LOBSTEROUT_CONTENT)
        filenames = extractor.fatband_filenames
        assert "FATBAND_Fe_s.lobster" in filenames
        assert "FATBAND_O_s.lobster" in filenames

    def test_mapping_protocol(self):
        extractor = LobsterOut.from_str(LOBSTEROUT_CONTENT)
        assert "ions_list" in extractor
        assert len(extractor) == 4
        assert extractor["ions_list"] == ["Fe", "O"]
