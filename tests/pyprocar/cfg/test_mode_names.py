import pytest

from pyprocar.cfg.band_structure import BandStructureMode as ConfigBandStructureMode
from pyprocar.cfg.unfold import UnfoldPlotMode
from pyprocar.scripts.scriptBandsplot import BandStructureMode


@pytest.mark.parametrize("mode_enum", [UnfoldPlotMode, BandStructureMode, ConfigBandStructureMode])
def test_scatter_mode_is_spelled_scatter_and_the_old_spelling_still_resolves(mode_enum):
    assert mode_enum("scatter").name == "SCATTER"
    assert mode_enum["SACATTER"] is mode_enum.SCATTER
    assert mode_enum(mode_enum.SACATTER) is mode_enum.SCATTER
    assert "SACATTER" not in [member.name for member in mode_enum]
