"""matplotlib >= 3.9 removed matplotlib.cm.get_cmap; make sure pyprocar
no longer relies on it."""

import re
from pathlib import Path
from types import SimpleNamespace

import pyprocar
from pyprocar.plotter.dos_plot import DOSPlot

PACKAGE_DIR = Path(pyprocar.__file__).parent


def test_no_removed_cm_get_cmap_calls():
    pattern = re.compile(r"\bcm\.get_cmap\(")
    offenders = []
    for filepath in PACKAGE_DIR.rglob("*.py"):
        for lineno, line in enumerate(filepath.read_text().splitlines(), start=1):
            if pattern.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{filepath.relative_to(PACKAGE_DIR)}:{lineno}")
    assert offenders == []


def test_dos_plot_bar_colors():
    fake_self = SimpleNamespace(config=SimpleNamespace(cmap="viridis"))

    colors = DOSPlot._get_bar_color(fake_self, [0.0, 0.5, 1.0])

    assert len(colors) == 3
    assert all(len(color) == 4 for color in colors)
