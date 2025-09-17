

NON_SPIN_POLARIZED_DIR = (
    ROOT_DIR / "tests" / "data" / "examples" / "bands" / "non-spin-polarized"
)
SPIN_POLARIZED_DIR = (
    ROOT_DIR / "tests" / "data" / "examples" / "bands" / "spin-polarized"
)
NON_COLINEAR_DIR = ROOT_DIR / "tests" / "data" / "examples" / "bands" / "non-colinear"


DOS_NON_SPIN_POLARIZED_DIR = (
    ROOT_DIR / "tests" / "data" / "examples" / "dos" / "non-spin-polarized"
)
DOS_SPIN_POLARIZED_DIR = (
    ROOT_DIR / "tests" / "data" / "examples" / "dos" / "spin-polarized"
)
DOS_NON_COLINEAR_DIR = ROOT_DIR / "tests" / "data" / "examples" / "dos" / "non-colinear"


GAMMA_POINT_DIR = ROOT_DIR / "tests" / "data" / "examples" / "bands" / "atomic_levels" / "hBN-C2"

from pyprocar.core.ebs import ElectronicBandStructurePath

ebs = ElectronicBandStructurePath.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)

from pyprocar.plotter.bs_plots import (
    AtomicLevelsPlot,
    BandsPlotter,
    OverlayPlot,
    ParametricPlot,
    QuiverPlot,
    ScatterPlot,
)


def test_bsplot_scatter(ebs: ElectronicBandStructurePath):
    projection_weights = ebs.compute_projected_sum(atoms=[1], orbitals=[4,5,6,7,8])
    bands_velocity = ebs.get_property("bands_velocity")

    p = BandsPlotter(scatter_kwargs={"s": 2})
    p.scatter(ebs.kpath, ebs.bands, scalars=projection_weights)
    p.show()

test_bsplot_scatter(ebs)


def test_bsplot_quiver(ebs: ElectronicBandStructurePath):
    bands_velocity = ebs.get_property("bands_velocity")

    p = BandsPlotter(scatter_kwargs={"s": 2})
    p.quiver(ebs.kpath, ebs.bands, vectors=bands_velocity)
    p.show()

test_bsplot_quiver(ebs)




def test_bsplot_overlay_species(ebs: ElectronicBandStructurePath):
    weights, labels = ebs.build_overlay_species_weights(orbitals=[4,5,6,7,8])
    p = BandsPlotter()
    p.overlay(ebs.kpath, ebs.bands, weights, labels=labels)
    p.show()

test_bsplot_overlay_species(ebs)

def test_bsplot_overlay_orbitals(ebs: ElectronicBandStructurePath):
    weights, labels = ebs.build_overlay_orbitals_weights(atoms=[2,3,4])
    p = BandsPlotter()
    p.overlay(ebs.kpath, ebs.bands, weights, labels=labels)
    p.show()

test_bsplot_overlay_orbitals(ebs)



# # generic overlay
def test_bsplot_overlay_generic(ebs: ElectronicBandStructurePath):
    items= {"V":[4,5,6,7,8]}
    weights, labels = ebs.build_overlay_weights(items, orbitals_as_names=True)
    p = BandsPlotter()
    p.overlay(ebs.kpath, ebs.bands, weights, labels=labels)
    p.show()

test_bsplot_overlay_generic(ebs)




def test_bsplot_parametric(ebs: ElectronicBandStructurePath):
    projection_weights = ebs.compute_projected_sum(atoms=[1], orbitals=[4,5,6,7,8])
    p = BandsPlotter()
    p.parametric(ebs.kpath, ebs.bands, scalars=projection_weights)
    p.show()

test_bsplot_parametric(ebs)


def test_bsplot_multi_method_call(ebs: ElectronicBandStructurePath):
    projection_weights = ebs.compute_projected_sum(atoms=[1], orbitals=[4,5,6,7,8])
    bands_velocity = ebs.get_property("bands_velocity")

    p = BandsPlotter(scatter_kwargs={"s": 2})
    p.scatter(ebs.kpath, ebs.bands, scalars=projection_weights)
    p.quiver(ebs.kpath, ebs.bands, vectors=bands_velocity)
    p.show()

test_bsplot_multi_method_call(ebs)
