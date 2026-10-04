import dataclasses
import json
import re

import pytest

from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.cfg.unfold import UnfoldingConfig
from tests.utils import ROOT_DIR

NOTEBOOKS = sorted((ROOT_DIR / "examples" / "00-band_structure").glob("*.ipynb"))
CONFIG_OF_CALL = {"pyprocar.bandsplot(": BandStructureConfig, "pyprocar.unfold(": UnfoldingConfig}
OPTION_LINE = re.compile(r"^(\w+) : ", re.MULTILINE)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_saved_outputs_name_only_current_plot_options(notebook):
    stale = []
    for index, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
        source = "".join(cell["source"])
        for call, config in CONFIG_OF_CALL.items():
            if call not in source:
                continue
            fields = {field.name for field in dataclasses.fields(config)}
            text = "".join("".join(out.get("text", "")) for out in cell.get("outputs", []))
            stale += [(index, name) for name in OPTION_LINE.findall(text) if name not in fields]

    assert stale == []
