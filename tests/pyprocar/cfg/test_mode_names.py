import pytest

from pyprocar.cfg.band_structure import BandStructureMode as ConfigBandStructureMode
from pyprocar.cfg.unfold import UnfoldPlotMode
from pyprocar.scripts.scriptBandsplot import BandStructureMode


@pytest.mark.parametrize("mode_enum", [UnfoldPlotMode, BandStructureMode, ConfigBandStructureMode])
def test_scatter_mode_member_is_spelled_scatter(mode_enum):
    assert mode_enum("scatter").name == "SCATTER"
    assert "SACATTER" not in mode_enum.__members__
