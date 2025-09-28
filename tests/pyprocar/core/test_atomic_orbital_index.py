from pyprocar.core.atomic_orbital_index import (
    AtomIndexer,
    OrbitalIndexer,
    ProjectionLabelBuilder,
    ProjectionSelectionResolver,
    SpinIndexer,
)


def test_atom_indexer_compacts_ranges():
    indexer = AtomIndexer(["F", "F", "F", "O", "O", "Sr", "Sr", "F", "Sr", "F"])
    label = indexer.label(indices=[0, 1, 2, 3, 4, 5, 6, 7, 9])
    assert label == "F_{0-2,7,9}O_{3-4}Sr_{5-6}"


def test_atom_indexer_respects_limit():
    indexer = AtomIndexer(
        ["F", "F", "F", "O", "O", "Sr", "Sr", "F", "Sr", "F"],
        max_indices_for_ranges=3,
    )
    label = indexer.label(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert label == "F,O,Sr"


def test_orbital_indexer_groups_primary_sets():
    indexer = OrbitalIndexer()
    label = indexer.label(indices=[1, 2, 3, 7])
    assert label == "p,d_xz"


def test_orbital_indexer_latex_tokens():
    indexer = OrbitalIndexer()
    _, latex = indexer.label_with_latex(indices=[6])
    assert latex == "d_{z^2}"

    custom_orbitals = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz"]
    _, custom_latex = indexer.label_with_latex(indices=[6], orbital_names=custom_orbitals)
    assert custom_latex == "d_{z^2}"

    _, dxy_latex = indexer.label_with_latex(indices=[4], orbital_names=custom_orbitals)
    assert dxy_latex == "d_{xy}"


def test_projection_label_builder_combines_sections():
    atom_indexer = AtomIndexer(["V", "V", "O"])
    spin_indexer = SpinIndexer.from_projection_names(["total", "Spin-up", "Spin-down"])
    builder = ProjectionLabelBuilder(
        atom_indexer=atom_indexer,
        spin_indexer=spin_indexer,
    )
    orbital_names = [
        "s",
        "p_y",
        "p_z",
        "p_x",
        "d_{xy}",
        "d_{yz}",
        "d_{z^2}",
        "d_{xz}",
    ]
    components = builder.build_components(
        atoms=[1],
        orbitals=[1, 2, 3, 7],
        spins=[1],
        species=["V"],
        orbital_names=orbital_names,
        include_spins=True,
        is_non_colinear=False,
    )
    assert components.combined == "V_{1}-(p,d_xz)[Spin-up]"
    assert components.combined_latex == "\\mathrm{V}_{1}-(p,d_{xz})[\\uparrow]"
    assert components.atom == "V_{1}"
    assert components.atom_latex == "\\mathrm{V}_{1}"
    assert components.orbital == "p,d_xz"
    assert components.orbital_latex == "p,d_{xz}"
    assert components.spin == "Spin-up"
    assert components.spin_latex == "\\uparrow"
    assert components.species == "V"
    assert components.species_latex == "\\mathrm{V}"


def test_projection_selection_resolver_resolves_species_selection():
    atom_indexer = AtomIndexer(["F", "F", "F", "O", "O", "Sr", "Sr", "F", "Sr", "F"])
    spin_indexer = SpinIndexer.from_projection_names(["total", "Spin-up"])
    orbital_indexer = OrbitalIndexer()
    builder = ProjectionLabelBuilder(
        atom_indexer=atom_indexer,
        orbital_indexer=orbital_indexer,
        spin_indexer=spin_indexer,
    )
    resolver = ProjectionSelectionResolver(
        label_builder=builder,
        orbital_names=["s", "p_y", "p_z", "p_x", "d_{xy}", "d_{yz}", "d_{z^2}", "d_{xz}"],
        is_non_colinear=False,
    )

    result = resolver.resolve(
        species=["F", "Sr"],
        orbitals=[1, 2, 3, 7],
        spins=[1],
    )

    assert result.atoms == (0, 1, 2, 5, 6, 7, 8, 9)
    assert result.species == ("F", "Sr")
    assert result.orbitals == (1, 2, 3, 7)
    assert result.spins == (1,)
    assert result.labels.combined.startswith("F_{0-2,7,9}Sr_{5-6,8}")
    assert result.labels.combined_latex.startswith("\\mathrm{F}_{0-2,7,9}\\mathrm{Sr}_{5-6,8}")


def test_projection_selection_resolver_species_orbital_map():
    atom_indexer = AtomIndexer(["Sr", "V", "O", "O", "O"])
    spin_indexer = SpinIndexer.from_projection_names(["total"])
    builder = ProjectionLabelBuilder(
        atom_indexer=atom_indexer,
        spin_indexer=spin_indexer,
    )
    resolver = ProjectionSelectionResolver(
        label_builder=builder,
        orbital_names=None,
        is_non_colinear=False,
    )

    species_orbital_map = [{"Sr": [0], "V": [1, 2, 3]}]
    result = resolver.resolve(spins=[0], species_orbital_map=species_orbital_map)

    assert result.species == ("Sr", "V")
    assert result.atoms == (0, 1)
    assert result.orbitals == (0, 1, 2, 3)
    assert result.labels.combined_latex.startswith("\\mathrm{Sr}_{0}\\mathrm{V}_{1}")


def test_projection_selection_resolver_defaults_to_all_atoms():
    atom_indexer = AtomIndexer(["Sr", "V", "O", "O", "O"])
    spin_indexer = SpinIndexer.from_projection_names(["total"])
    builder = ProjectionLabelBuilder(
        atom_indexer=atom_indexer,
        spin_indexer=spin_indexer,
    )
    resolver = ProjectionSelectionResolver(
        label_builder=builder,
        orbital_names=None,
        is_non_colinear=False,
    )

    result = resolver.resolve()

    assert result.atoms == (0, 1, 2, 3, 4)
    assert set(result.species) == {"Sr", "V", "O"}
