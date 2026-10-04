import numpy as np

from pyprocar.io.procarparser import ProcarParser

ORBITALS = "ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot"


def _band_block(header: str, weights: tuple[float, float]) -> list[str]:
    rows = [
        f"    {ion}  {w:.3f}" + "  0.000" * 8 + f"  {w:.3f}" for ion, w in enumerate(weights, 1)
    ]
    total = sum(weights)
    rows.append(f"tot    {total:.3f}" + "  0.000" * 8 + f"  {total:.3f}")
    return [
        header,
        "",
        "band     1 # energy  -1.00000000 # occ.  1.00000000",
        " ",
        ORBITALS,
        *rows,
        "",
    ]


def test_kpoint_coordinates_without_a_separating_space_are_read(tmp_path):
    headers = [
        " k-point     1 :    0.00000000 0.00000000 0.00000000     weight = 0.50000000",
        " k-point     2 :    0.50000000-0.25000000 0.00000000     weight = 0.50000000",
    ]
    lines = [
        "PROCAR lm decomposed",
        "# of k-points:  2         # of bands:   1         # of ions:    2",
        "",
    ]
    lines += _band_block(headers[0], (0.1, 0.2)) + _band_block(headers[1], (0.3, 0.4))
    path = tmp_path / "PROCAR"
    path.write_text("\n".join(lines) + "\n")

    procar = ProcarParser()
    procar.readFile(str(path))

    assert np.asarray(procar.kpoints).tolist() == [[0.0, 0.0, 0.0], [0.5, -0.25, 0.0]]
    assert np.asarray(procar.spd)[1, 0, 0, :, 1].tolist() == [0.3, 0.4, 0.7]
