from pyprocar.cfg import ConfigManager, PlotType
from pyprocar.cfg.band_structure import BandStructureConfig


def test_merge_configs_leaves_default_untouched():
    default = BandStructureConfig(plot_type=PlotType.BAND_STRUCTURE)

    merged = ConfigManager.merge_configs(
        default, {"fermi_linewidth": 7.5, "unknown_key": 1}
    )

    assert isinstance(merged, BandStructureConfig)
    assert merged.fermi_linewidth == 7.5
    assert merged.custom_settings == {"unknown_key": 1}
    assert default.fermi_linewidth == 1
    assert default.custom_settings == {}
