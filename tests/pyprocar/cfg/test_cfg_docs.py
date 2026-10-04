import ast
import dataclasses
import re

import pytest

from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.band_structure_2d import Bandstructure2DConfig
from pyprocar.cfg.base import PlotType
from pyprocar.cfg.dos import DensityOfStatesConfig
from pyprocar.cfg.fermi_surface_2d import FermiSurface2DConfig
from pyprocar.cfg.fermi_surface_3d import FermiSurface3DConfig
from pyprocar.cfg.unfold import UnfoldingConfig
from tests.utils import ROOT_DIR

CFG_DOCS = ROOT_DIR / "docs" / "source" / "api" / "cfg"

CONFIGS = {
    "band_structure": (BandStructureConfig, PlotType.BAND_STRUCTURE),
    "band_structure_2d": (Bandstructure2DConfig, PlotType.BAND_STRUCTURE_2D),
    "dos": (DensityOfStatesConfig, PlotType.DENSITY_OF_STATES),
    "fermi_surface_2d": (FermiSurface2DConfig, PlotType.FERMI_SURFACE_2D),
    "fermi_surface_3d": (FermiSurface3DConfig, PlotType.FERMI_SURFACE_3D),
    "unfold": (UnfoldingConfig, PlotType.UNFOLD),
}

DOCUMENTED_DEFAULT = re.compile(
    r"^\s*(?P<name>\w+)\s*:[^\n]*\(default (?P<inline>[^\n]+)\)\s*$"
    r"|^\s*(?P<name2>\w+)\s*:[^\n]*\n[^\n]*default is (?P<prose>[^\n]+?)\.?\s*$",
    re.MULTILINE,
)


def _as_list(value):
    return list(value) if isinstance(value, tuple | list) else value


def test_cfg_docs_name_only_existing_files():
    named = {
        path
        for page in CFG_DOCS.glob("*.rst")
        for path in re.findall(r"``(pyprocar/cfg/[^`]+)``", page.read_text())
    }

    assert sorted(path for path in named if not (ROOT_DIR / path).exists()) == []


@pytest.mark.parametrize("module", [m for m in CONFIGS if m != "fermi_surface_2d"])
def test_cfg_page_renders_the_config_module(module):
    assert f".. automodule:: pyprocar.cfg.{module}" in (CFG_DOCS / f"{module}.rst").read_text()


@pytest.mark.parametrize("module", CONFIGS)
def test_documented_config_defaults_match_the_config(module):
    config_class, plot_type = CONFIGS[module]
    config = config_class(plot_type=plot_type)
    fields = {field.name for field in dataclasses.fields(config_class)}

    stale = []
    for match in DOCUMENTED_DEFAULT.finditer(config_class.__doc__ or ""):
        name = match["name"] or match["name2"]
        documented = (match["inline"] or match["prose"]).strip()
        if name not in fields:
            stale.append((name, documented, "not a field"))
            continue
        try:
            value = ast.literal_eval(documented)
        except (ValueError, SyntaxError):
            stale.append((name, documented, "not a Python literal"))
            continue
        if _as_list(value) != _as_list(getattr(config, name)):
            stale.append((name, documented, getattr(config, name)))

    assert stale == []
