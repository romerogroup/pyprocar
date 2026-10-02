# Drives a plain band structure through the new object API (dos-rewrite branch):
# ElectronicBandStructurePath.from_code -> BandStructurePlotter.plot -> PNG.
# Harness env: CALC (copied calc dir), EVIDENCE (output dir).
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from pyprocar.core.ebs import ElectronicBandStructurePath
from pyprocar.plotter.bs_plot import BandStructurePlotter

calc, evidence = Path(os.environ["CALC"]), Path(os.environ["EVIDENCE"])
png = evidence / "bands_plain.png"

ebs = ElectronicBandStructurePath.from_code(code="vasp", dirpath=calc)
p = BandStructurePlotter()
p.plot(ebs.bands, scalars_mode="none")
fig = plt.gcf()
fig.savefig(png, dpi=150)

ax = fig.axes[0]
summary = {
    "bands_shape": list(ebs.bands.to_array().shape),
    "n_lines": len(ax.get_lines()) + len(ax.collections),
    "ylim": [round(v, 3) for v in ax.get_ylim()],
    "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
    "png_bytes": png.stat().st_size,
}
(evidence / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
assert summary["n_lines"] > 0 and summary["png_bytes"] > 10_000
