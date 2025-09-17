
from pyprocar.core import BandStructure2D
from pyprocar.core.bandstructure2D import BandStructure2D

####################################################################################################
# Electronic 3d mesh slice plotting
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from pyprocar.plotter.ebs_plane_plot import EBSPlanePlotter

# Example 1: Bands and Bands Velocity
ebs = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
iband=2
ispin=0
bands_velocity = ebs.get_property("bands_velocity").value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20))
p.plot_scalars(("bands", ebs.bands[:, iband, ispin]))
p.plot_vectors_quiver(("bands_velocity", bands_velocity[:, iband, ispin, :]))
p.show_colorbar()
p.show()

# Example 2: Bands Speed
ebs = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
ebs.reduce_bands_near_fermi()
iband=2
ispin=0
bands_speed = ebs.get_property("bands_speed").value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20))
p.plot_scalars(("bands_speed", bands_speed[:, iband, ispin]), cmap="plasma")
p.show_colorbar()
p.show()

# Example 3: Avg Inverse Effective Mass
ebs = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
ebs.reduce_bands_near_fermi()
iband=2
ispin=0
avg_inv_effective_mass = ebs.get_property("avg_inv_effective_mass").value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20))
p.plot_scalars(("avg_inv_effective_mass", avg_inv_effective_mass[:, iband, ispin]), cmap="plasma")
p.show_colorbar()
p.show()

# Example 4: EBS IPR
ebs = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
ebs.reduce_bands_near_fermi()
iband=2
ispin=0
ebs_ipr = ebs.get_property("ebs_ipr").value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20))
p.plot_scalars(("ebs_ipr", ebs_ipr[:, iband, ispin]), cmap="plasma")
p.show_colorbar()
p.show()

# Example 5: projected sum
ebs = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
ebs.reduce_bands_near_fermi()
iband=2
ispin=0
projected_sum = ebs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8]).value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20))
p.plot_scalars(("projected_sum", projected_sum[:, iband, ispin]), cmap="plasma")
p.show_colorbar()
p.show()


# Example 6: Spin Texture

BiSB_DIR = ROOT_DIR / "tests" / "data" / "examples" / "fermi2d" / "bisb_monolayer"

ebs = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=BiSB_DIR)
ebs.reduce_bands_near_energy(energy=ebs.fermi+0.9)
iband=0
ispin=0
spin_texture = ebs.get_property("projected_sum_spin_texture", atoms=[1], orbitals=[4, 5, 6, 7, 8]).value
ebs.compute_gradients(gradient_order=2, names=["projected_sum_spin_texture"])



# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# # 2.1 Property
spin_texture = ebs.get_property("projected_sum_spin_texture", atoms=[1], orbitals=[4, 5, 6, 7, 8]).value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20), ax=axes[0, 0])
p.plot_vectors_quiver(("spin_texture", spin_texture[:, iband, ispin, :]), cmap="coolwarm", clim=(-0.5, 0.5))
p.show_colorbar()
axes[0, 0].set_title("Spin Texture")
# p.show()

# # 2.6 laplacian
spin_texture = ebs.get_property("projected_sum_spin_texture", atoms=[1], orbitals=[4, 5, 6, 7, 8]).value
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20), ax=axes[1, 0])
p.plot_vectors_quiver(("spin_texture", spin_texture[:, iband, ispin, :]), cmap="coolwarm", clim=(-0.5, 0.5), plot_scalar=True)
p.show_colorbar()
axes[1, 0].set_title("Vector Field")
# p.show()


# # 2.2 Divergence
spin_texture_divergence = ebs.get_property("projected_sum_spin_texture").divergence

p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20), ax=axes[0, 1])
p.plot_scalars(("spin_texture_divergence", spin_texture_divergence[:, iband, ispin]), cmap="coolwarm", clim=(-0.5, 0.5))
p.show_colorbar()
axes[0, 1].set_title("Divergence")
# p.show()

# # 2.3 Divergence Gradient
spin_texture_divergence_gradient = ebs.get_property("projected_sum_spin_texture").divergence_gradient
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20), ax=axes[1, 1])
p.plot_vectors_quiver(("spin_texture_divergence_gradient", spin_texture_divergence_gradient[:, iband, ispin, :]), cmap="coolwarm", clim=(-0.5, 0.5), plot_scalar=True)
p.show_colorbar()
axes[1, 1].set_title("Divergence Gradient")
# p.show()

# # 2.4 Curl
spin_texture_curl = ebs.get_property("projected_sum_spin_texture").curl
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20), ax=axes[0, 2])
p.plot_vectors_quiver(("spin_texture_curl", spin_texture_curl[:, iband, ispin, :]), cmap="coolwarm", clim=(-0.5, 0.5), plot_scalar=True)
p.show_colorbar()
axes[0, 2].set_title("Curl")
# p.show()

# # 2.5 curl_gradient
spin_texture_curl_gradient = ebs.get_property("projected_sum_spin_texture").curl_gradient
p = EBSPlanePlotter(ebs, normal=(0, 0, 1), origin=(0, 0, 0), grid_interpolation=(20, 20), ax=axes[1, 2])
p.plot_vectors_quiver(("spin_texture_curl_gradient", spin_texture_curl_gradient[:, iband, ispin, :]), cmap="coolwarm", clim=(-0.5, 0.5), plot_scalar=True)
p.show_colorbar()
axes[1, 2].set_title("Curl Gradient")
# p.show()

plt.tight_layout()
plt.show()

###############################################################################################################################
####################################################################################################
# Fermi 3d plotter


fsplt = FermiPlotter()
fsplt.add_brillouin_zone(new_fs.brillouin_zone)
fsplt.add_surface(new_fs, show_scalar_bar=True)
fsplt.show()


fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()
fs.set_band_colors()
# fs.get_property("bands")
fs.plot()



# Example 1
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()


# Example 2
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("fermi_velocity")
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True, add_active_vectors=True)
fsplt.show()


# Example 3
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("fermi_speed")
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()

# Example 4
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("avg_inv_effective_mass")
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()


# Example 5
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("ebs_ipr")
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()


# Example 7
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8])
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()

# Example 8
fs = FermiSurface.from_code(code="vasp", dirpath=SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0])
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()

fs = FermiSurface.from_code(code="vasp", dirpath=SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[1])
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()

# Example 9
fs = FermiSurface.from_code(code="vasp", dirpath=NON_COLINEAR_DIR)
fs.get_property("projected_sum_spin_texture", atoms=[1], orbitals=[4, 5, 6, 7, 8])
fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True, add_active_vectors=True)
fsplt.show()



# Example 10: Fermi Cross Section

# 10.1: Scalar Value
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[1])
fsplt = FermiPlotter()
fsplt.add_slicer(fs, normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.show()

# 10.2: Vector Value
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("fermi_velocity")
fsplt = FermiPlotter()
fsplt.add_slicer(fs, normal=(0, 0, 1), origin=(0, 0, 0), add_active_vectors=True)
fsplt.show()


# Example 10: Fermi Cross Section Box Slicer

# 10.1: Scalar Value
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[1])
fsplt = FermiPlotter()
fsplt.add_box_slicer(fs, normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.show()

# 10.2: Vector Value
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("fermi_velocity")
fsplt = FermiPlotter()
fsplt.add_box_slicer(fs, normal=(0, 0, 1), origin=(0, 0, 0), add_active_vectors=True)
fsplt.show()



# Example 11: Fermi IsoSlider
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fsplt = FermiPlotter()
fermi = fs.fermi
differences = [-0.2, 0.0, 0.2]
fermi_differences = [fermi + difference for difference in differences]

fss = []

for fermi_difference in fermi_differences:
    fs = FermiSurface.from_code(
        code="vasp", dirpath=NON_SPIN_POLARIZED_DIR, fermi=fermi_difference
    )
    fs.get_property("fermi_velocity")
    fss.append(fs)

fsplt.add_isoslider(fss, fermi_differences, add_active_vectors=True)
fsplt.show()


# Example 12: Fermi IsoSlider

fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fsplt = FermiPlotter()
fermi = fs.fermi
differences = [-0.2, 0.0, 0.2]
fermi_differences = [fermi + difference for difference in differences]

fss = []

for fermi_difference in fermi_differences:
    fs = FermiSurface.from_code(
        code="vasp", dirpath=NON_SPIN_POLARIZED_DIR, fermi=fermi_difference
    )
    fs.get_property("fermi_velocity")
    fss.append(fs)
fsplt.add_isovalue_gif(
    fss,
    save_gif=CURRENT_DIR / "test.gif",
    iter_reverse=True,
    add_active_vectors=True,
)



######################################################################################################
# # Fermi 2d plotter
# # 

# Example 1
fs = FermiSurface.from_code(code="vasp", dirpath=NON_COLINEAR_DIR)
fs.get_property("projected_sum_spin_texture", atoms=[1], orbitals=[4, 5, 6, 7, 8])
fsplt = FermiSlicePlotter(normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.plot(fs)
fsplt.plot_arrows(fs)
fsplt.show()

# Example 2
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.get_property("fermi_velocity")
extended_fs = fs.extend_surface(zone_directions=[(1, 0, 0), 
                                   (0, 1, 0), 
                                   (-1, 0, 0), 
                                   (0, -1, 0),
                                   (1,-1,0),
                                   (-1,1,0),
                                   (1, 1, 0), 
                                   (-1, -1, 0)])
fsplt = FermiSlicePlotter(normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.plot(extended_fs)
fsplt.plot_arrows(extended_fs)
fsplt.show()

# Example 3

# #3.1
fs = FermiSurface.from_code(code="vasp", dirpath=SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[0])
fsplt = FermiSlicePlotter(normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.plot(fs)
fsplt.show()

# #3.2
fs = FermiSurface.from_code(code="vasp", dirpath=SPIN_POLARIZED_DIR)
fs.get_property("projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], spins=[1])
fsplt = FermiSlicePlotter(normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.plot(fs)
fsplt.show_colorbar()
fsplt.show()



# # Example 4
fs = FermiSurface.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
fs.compute_atomic_projection(atoms=[1], orbitals=[4, 5, 6, 7, 8])
fsplt = FermiSlicePlotter(normal=(0, 0, 1), origin=(0, 0, 0))
fsplt.scatter(fs)
fsplt.show()




####################################################################################################
# BandStructure2D



ebs_mesh = ElectronicBandStructureMesh.from_code(code="vasp", dirpath=NON_SPIN_POLARIZED_DIR)
bs2d = BandStructure2D(ebs=ebs_mesh)


bs2d = BandStructure2D.from_code(code="vasp", 
                                 dirpath=GRAPHENE_DIR, 
                                 normal=(0, 0, 1), 
                                #  normal=(0, 1, 1), 
                                 origin=(0, 0, 0.0), 
                                 grid_interpolation=(100, 100),
                                 padding=20,
                                 as_cartesian=False,
                                 reduce_bands_near_fermi=True)

# # bs2d.get_property("avg_inv_effective_mass")

# bs2d.get_property("bands_velocity")

bsplt = BS2DPlotter(bs2d)
bz = bs2d.get_2d_brillouin_zone(e_min=-2, e_max=2.5)

# print(bs2d.point_data)
bsplt.add_brillouin_zone(bz)
bsplt.show_grid()
bsplt.add_surface(bs2d, add_active_vectors=True)
bsplt.show()




fsplt = FermiPlotter()
fsplt.add_brillouin_zone(fs.brillouin_zone)
fsplt.add_surface(fs, show_scalar_bar=True)
fsplt.show()
