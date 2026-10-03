from pyprocar.cfg import ConfigManager, PlotType
from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.unfold import UnfoldingConfig
from tests.utils import ROOT_DIR


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


def test_unfold_config_has_no_options_without_a_reader():
    options = UnfoldingConfig(plot_type=PlotType.UNFOLD).as_dict()

    assert {"modes", "weighted_color", "weighted_width"} & options.keys() == set()
    assert options["cmap"] == "jet"
    assert not (ROOT_DIR / "pyprocar" / "cfg" / "unfold.yml").exists()
