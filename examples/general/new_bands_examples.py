import logging
import time

start_time = time.time()
import os
from pathlib import Path

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)


from dotenv import load_dotenv

load_dotenv()


print(os.getenv("DATA_DIR"))
DATA_DIR = Path(os.getenv("DATA_DIR"))


NON_SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "bands" / "non-spin-polarized"
SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "bands" / "spin-polarized"
NON_COLINEAR_DIR = DATA_DIR / "examples" / "bands" / "non-colinear"


DOS_NON_SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "dos" / "non-spin-polarized"
DOS_SPIN_POLARIZED_DIR = DATA_DIR / "examples" / "dos" / "spin-polarized"
DOS_NON_COLINEAR_DIR = DATA_DIR / "examples" / "dos" / "non-colinear"


GAMMA_POINT_DIR = DATA_DIR / "examples" / "bands" / "atomic_levels" / "hBN-C2"

# Results directory for saving plots
RESULTS_DIR = Path(__file__).parent / "results" / "bands"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from pyprocar.core.ebs import ElectronicBandStructurePath
from pyprocar.plotter.bs_plot import BandStructurePlotter


def save_plot(name: str):
    """Save the current plot to RESULTS_DIR and close the figure."""
    plt.tight_layout()
    output_path = RESULTS_DIR / f"{name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


ebs = ElectronicBandStructurePath.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)


def test_bsplot_plain(ebs: ElectronicBandStructurePath):
    """Plot plain band structure using .plot() with scalars_mode='none'."""
    p = BandStructurePlotter()
    p.plot(ebs.bands, scalars_mode="none")
    save_plot("test_bsplot_plain")


def test_bsplot_scatter(ebs: ElectronicBandStructurePath):
    """Plot scatter band structure using .plot() with scalars_mode='scatter'."""
    projection_weights = ebs.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8])

    p = BandStructurePlotter()
    p.plot(ebs.bands, scalars_data=projection_weights, scalars_mode="scatter", scatter_kwargs={"s": 2, "c": "red"})
    save_plot("test_bsplot_scatter")


def test_bsplot_quiver(ebs: ElectronicBandStructurePath):
    """Plot quiver band structure using plot_quiver() (specialized method for vectors)."""
    bands_velocity = ebs.get_property("bands_velocity")

    p = BandStructurePlotter()
    p.plot(ebs.bands, vectors_data=bands_velocity, vectors_mode="quiver")
    save_plot("test_bsplot_quiver")


def test_bsplot_overlay_species(ebs: ElectronicBandStructurePath):
    """Plot overlay by species using plot_overlay() (specialized method for multi-weight fills)."""
    properties = ebs.build_overlay_species_weights(orbitals=[4, 5, 6, 7, 8])
    weights = [prop.to_array() for prop in properties]
    labels = [prop.label for prop in properties]
    p = BandStructurePlotter()
    p.plot_overlay(ebs.kpath, ebs.bands, weights, labels=labels)
    save_plot("test_bsplot_overlay_species")


def test_bsplot_overlay_orbitals(ebs: ElectronicBandStructurePath):
    """Plot overlay by orbitals using plot_overlay() (specialized method for multi-weight fills)."""
    properties = ebs.build_overlay_orbitals_weights(atoms=[2, 3, 4])
    weights = [prop.to_array() for prop in properties]
    labels = [prop.label for prop in properties]
    p = BandStructurePlotter()
    p.plot_overlay(ebs.kpath, ebs.bands, weights, labels=labels)
    save_plot("test_bsplot_overlay_orbitals")


def test_bsplot_overlay_generic(ebs: ElectronicBandStructurePath):
    """Plot overlay with generic items using plot_overlay() (specialized method for multi-weight fills)."""
    items = {"V": [4, 5, 6, 7, 8]}
    properties = ebs.build_overlay_weights(items)
    weights = [prop.to_array() for prop in properties]
    labels = [prop.label for prop in properties]
    p = BandStructurePlotter()
    p.plot_overlay(ebs.kpath, ebs.bands, weights, labels=labels)
    save_plot("test_bsplot_overlay_generic")


def test_bsplot_parametric(ebs: ElectronicBandStructurePath):
    """Plot parametric band structure using .plot() with scalars_mode='parametric'."""
    projection_weights = ebs.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8])
    p = BandStructurePlotter()
    p.plot(ebs.bands, scalars_data=projection_weights, scalars_mode="parametric")
    save_plot("test_bsplot_parametric")


def test_bsplot_multi_method_call(ebs: ElectronicBandStructurePath):
    """Demonstrate chaining multiple specialized plot methods on the same plotter."""
    projection_weights = ebs.compute_projected_sum(atoms=[1], orbitals=[4, 5, 6, 7, 8])
    bands_velocity = ebs.get_property("bands_velocity")

    p = BandStructurePlotter()
    p.plot_scatter(ebs.kpath, ebs.bands, scalars=projection_weights.to_array(), s=2)
    p.plot_quiver(ebs.kpath, ebs.bands, vectors=bands_velocity.to_array())
    save_plot("test_bsplot_multi_method_call")


###########################################################
# Run tests
###########################################################
# Tests using unified .plot() method with scalars_mode
test_bsplot_plain(ebs)
test_bsplot_scatter(ebs)
test_bsplot_parametric(ebs)

# Tests using specialized methods (quiver, overlay)
test_bsplot_quiver(ebs)
test_bsplot_overlay_species(ebs)
test_bsplot_overlay_orbitals(ebs)  # Bug in build_overlay_orbitals_weights
test_bsplot_overlay_generic(ebs)  # Bug in build_overlay_weights
test_bsplot_multi_method_call(ebs)

print(f"Time taken: {time.time() - start_time} seconds")
