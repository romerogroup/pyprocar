

class ParametricPlot(BasePlotter):

    def _plot(self, energies,  ,**kwargs):
        values_dict = {}
        spin_projections, spin_channels = self._get_spins_projections_and_channels(
            spins
        )
        dos_total, dos_total_projected, dos_projected = self._calculate_parametric_dos(
            atoms, orbitals, spin_projections, principal_q_numbers
        )

        orbital_string = ":".join([str(orbital) for orbital in orbitals])
        atom_string = ":".join([str(atom) for atom in atoms])
        spin_string = ":".join(
            [str(spin_projection) for spin_projection in spin_projections]
        )

        self._setup_colorbar(dos_projected, dos_total_projected)
        self._set_plot_limits(spin_channels)

        for ispin, spin_channel in enumerate(spin_channels):
            energies, dos_spin_total, normalized_dos_spin_projected = (
                self._prepare_parametric_spin_data(
                    spin_channel, ispin, dos_total, dos_projected, dos_total_projected,
                    scale=kwargs.get("scale", False)
                )
            )

            self._plot_spin_data_parametric(
                energies, dos_spin_total, normalized_dos_spin_projected
            )

            if self.config.plot_total:
                self._plot_total_dos(energies, dos_spin_total, spin_channel)

            values_dict["energies"] = energies
            values_dict["dosTotalSpin-" + str(spin_channel)] = dos_spin_total
            values_dict[
                "spinChannel-"
                + str(spin_channel)
                + f"_orbitals-{orbital_string}"
                + f"_atoms-{atom_string}"
                + f"_spinProjection-{spin_string}"
            ] = normalized_dos_spin_projected

        self.values_dict = values_dict
        return values_dict