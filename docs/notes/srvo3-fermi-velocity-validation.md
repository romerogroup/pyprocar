# SrVO3 Fermi velocity validation

This note records the external evidence that pyprocar's Fermi-surface band speeds for SrVO3 are physically correct after the reciprocal-lattice 2π fix (PR #240).

## Property confirmed

The property is the bare DFT band speed on the Fermi surface, |v| = (1/ħ)|∇ₖE| in m/s, with k in angular units (rad/Å).
pyprocar exposes it as `fermi_speed` (and `fermi_velocity`) on a `FermiSurface`, and as `bands_speed` on an `ElectronicBandStructureMesh`.

"Bare" means the Kohn-Sham band velocity, with no many-body renormalisation.
Experiment (ARPES) measures a renormalised velocity, which for SrVO3 is about half the DFT value.

The note does not confirm the effective mass; see [What this does not confirm](#what-this-does-not-confirm).

## pyprocar values

SrVO3, cubic perovskite, a ≈ 3.85 Å, V t2g bands crossing E_F.

| Calculation | Before PR #240 | After PR #240 |
|---|---|---|
| VASP, `data/examples/fermi3d/non-spin-polarized`, colourbar range | 2.41e6 to 3.05e6 m/s | 3.83e5 to 4.85e5 m/s |
| VASP, band 16 median on the surface | 2.78e6 m/s | 4.42e5 m/s |
| QE 7.2, `data/codes/qe/7.2/SrVO3/non-spin-polarized/fermi`, colourbar range | not loadable before #240 | 3.96e5 to 4.86e5 m/s |
| QE, band 20 median on the surface | not loadable before #240 | 4.56e5 m/s |

The before values are exactly 2π too large.
An independent central difference on the raw eigenvalues, with physical k, gives head/reference = 1.0000003 (VASP) and 1.000008 (QE).

## External evidence

Conversion used throughout: v [m/s] = ħv [eV·Å] × 1.519e5.

| Source | Type | Quoted value | Converted | What it confirms | Status |
|---|---|---|---|---|---|
| T. Yoshida, M. Hashimoto, T. Takizawa, A. Fujimori et al., Phys. Rev. B 82, 085119 (2010), [arXiv:1003.2269](https://arxiv.org/abs/1003.2269) | ARPES | "we have deduced the Fermi velocity vF ∼ 1.7 and ∼ 1.4 eVÅ for SrVO3 and CaVO3, respectively" | 2.6e5 m/s (renormalised) | Measured scale of the Fermi velocity | Quote checked against the arXiv text |
| Same paper | ARPES vs LDA | "The observed band widths for both compounds are almost half of those predicted by local-density-approximation band-structure calculation" | Bare DFT ≈ 2 × 2.6e5 ≈ 5e5 m/s | DFT velocity is about twice the ARPES value | Quote checked; the 5e5 is inferred from it |
| Yoshida et al., Phys. Rev. Lett. 95, 146404 (2005), [arXiv:cond-mat/0504075](https://arxiv.org/abs/cond-mat/0504075) | ARPES | Mass renormalisation m*/m_b ≈ 1.8 ± 0.2 | (factor, no velocity) | Independent support for the ~2× DFT-to-ARPES ratio | Value checked against the arXiv text |
| Nearest-neighbour t2g tight binding (this note) | Model | ε = −2t(cos kₓa + cos k_ya) for the xy band, t = 0.26 to 0.30 eV, a = 3.84 Å, 1/3 filling | 3.0e5 to 4.2e5 m/s on the Fermi surface | Expected magnitude of a bare t2g velocity | Our calculation; t is an assumed typical LDA value, not taken from a cited source |

Journal references and authors are taken from the arXiv metadata.

### Tight-binding estimate

The xy band's in-plane speed is (2ta/ħ)·√(sin²kₓa + sin²k_ya), with maximum 2√2·ta/ħ.
For t = 0.30 eV and a = 3.84 Å, 2ta/ħ ≈ 3.5e5 m/s.
Sampling the 1/3-filled Fermi line gives speeds of 3.5e5 to 4.2e5 m/s (median 3.8e5).
Real LDA bands add longer-range hopping and t2g-eg hybridisation, which raise the speeds somewhat above this nearest-neighbour model.

## Conclusion

The corrected values (median 4.4e5 to 4.6e5 m/s, range 3.8e5 to 4.9e5) sit between the nearest-neighbour estimate (3.0e5 to 4.2e5) and twice the ARPES value (about 5e5).
That is consistent to within about 30%, which is the precision these references support.

The pre-fix values (2.4e6 to 3.05e6 m/s) are 5 to 10 times the ARPES value and 6 to 8 times the tight-binding maximum, so they are ruled out.
Their ratio to the corrected values is 2π, which matches the cause fixed in PR #240.

No reference found quotes a DFT Fermi velocity for SrVO3 directly in m/s or eV·Å, so this is not a DFT-to-DFT comparison.

## What this does not confirm

The effective mass m_e/m* = 0.115 at Γ for band 16 is not a physical t2g band mass, and no reference supports it.

At Γ the three t2g bands are degenerate, and pyprocar sorts bands by energy, so band 16 is the lowest of the three at every k.
Along any axis the lowest band is the one that barely disperses in that direction, so the curvature measures the small hopping t′ along that axis.
With t′ ≈ 0.03 eV, m_e/m* ≈ t′a² / (ħ²/2m_e) ≈ 0.03 × 14.7 / 3.81 ≈ 0.12, which matches the computed value.

PR #240 does correct the effective mass by exactly (2π)², which the analytic free-electron test confirms (m_e/m* = 1).
A physical SrVO3 mass check needs a point away from the degeneracy, or unsorted bands.

## Reproduce

- `tests/pyprocar/core/test_fermisurface.py::test_fermi_speed_matches_finite_difference` checks the SrVO3 band-16 median of 4.419e5 m/s against an independent central difference (needs `data/`).
- `tests/pyprocar/core/test_ebs.py::test_free_electron_band_has_analytic_velocity_and_unit_mass` and `::test_free_electron_band_along_a_path_has_analytic_speed` check v = ħk/m and m_e/m* = 1 on synthetic bands.
- The before and after Fermi-surface images are attached to PR #240.
