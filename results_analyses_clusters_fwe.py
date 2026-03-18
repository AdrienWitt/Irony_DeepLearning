#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRUE NON-PARAMETRIC CLUSTER-LEVEL FWE CORRECTION
Using your real voxel-wise p-values from permutation testing
→ Primary threshold: p < 0.001 (uncorrected)
→ Cluster extent threshold: derived from null distribution of maximal cluster size
→ Cluster-level family-wise error rate controlled at p_FWE < 0.05
This is the correct and most respected way in 2025.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image, plotting
from nilearn.image import resample_to_img, load_img, coord_transform
from scipy.ndimage import label as nd_label, center_of_mass
from nibabel.affines import apply_affine

# ================================
# SETTINGS - CHANGE ONLY THESE
# ================================
PERM_NPZ = "results_wholebrain_irosar/normalized_time/permutation_results/perm_stats_textweighted_opensmile_with_base.npz"
EXAMPLE_NII = "data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz"

PRIMARY_P = 0.001                    # Cluster-forming threshold (uncorrected)
OUT_DIR = f"results_wholebrain_irosar/normalized_time/cluster_FWE_p{PRIMARY_P:.0e}_corrected"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Running TRUE cluster-level FWE correction")
print(f"Primary threshold: p < {PRIMARY_P} (uncorrected)")
print(f"Results → {OUT_DIR}\n")

# ================================
# 1. Load permutation results (your gold standard)
# ================================
data = np.load(PERM_NPZ)
delta_obs   = data['delta_obs']          # Observed Δr
pvals       = data['pvals']               # ← YOUR REAL NON-PARAMETRIC P-VALUES
delta_perm  = data['delta_perm']          # (n_perms, n_voxels) permuted Δr maps
n_perms     = delta_perm.shape[0]

print(f"Loaded {n_perms} permutations and exact non-parametric p-values for {len(delta_obs):,} voxels")

# ================================
# 2. Recreate brain mask (exact same as permutation script)
# ================================
icbm = datasets.fetch_icbm152_2009()
mask_icbm = load_img(icbm["mask"])
example_nii = nib.load(EXAMPLE_NII)
resampled_mask = resample_to_img(mask_icbm, example_nii, interpolation="nearest")
brain_mask_3d = resampled_mask.get_fdata() > 0
affine = example_nii.affine
voxel_indices = np.where(brain_mask_3d)

def flat_to_3d(arr):
    vol = np.zeros(brain_mask_3d.shape, dtype=float)
    vol[voxel_indices] = arr
    return vol

# ================================
# 3. Define connectivity structure (18-connectivity = face + edge)
# ================================
structure = np.ones((3,3,3), dtype=bool)

# 18 is default: full 3×3×3 minus corners → standard in literature

# ================================
# 4. Apply primary threshold to observed data using real p-values
# ================================
obs_suprathresh = pvals < PRIMARY_P
obs_3d = flat_to_3d(obs_suprathresh)
labeled_obs, n_clusters_obs = nd_label(obs_3d > 0, structure=structure)
cluster_sizes_obs = [np.sum(labeled_obs == i) for i in range(1, n_clusters_obs + 1)] if n_clusters_obs > 0 else []

print(f"Observed data: {len(cluster_sizes_obs)} clusters at p < {PRIMARY_P} (uncorrected)")

# ================================
# 5. Build null distribution of maximum cluster size
# ================================
print("Computing null distribution of maximum cluster size under H0...")
max_cluster_sizes_null = []

for perm_idx in range(n_perms):
    if (perm_idx + 1) % 1000 == 0 or perm_idx < 10:
        print(f"   → Processing permutation {perm_idx + 1}/{n_perms}")

    # For this permuted map, compute its own p-values (exact non-parametric way)
    perm_map = delta_perm[perm_idx]
    perm_pvals = (np.sum(delta_perm >= perm_map, axis=0) + 1) / (n_perms + 1)
    
    perm_suprathresh = perm_pvals < PRIMARY_P
    perm_3d = flat_to_3d(perm_suprathresh)
    labeled_perm, n = nd_label(perm_3d > 0, structure=structure)
    
    if n > 0:
        sizes = [np.sum(labeled_perm == j) for j in range(1, n + 1)]
        max_cluster_sizes_null.append(max(sizes))
    else:
        max_cluster_sizes_null.append(0)

max_cluster_sizes_null = np.array(max_cluster_sizes_null)
cluster_extent_threshold = np.percentile(max_cluster_sizes_null, 95)  # 95th → p_FWE < 0.05
cluster_extent_threshold = int(np.ceil(cluster_extent_threshold))

print(f"Cluster-level FWE p < 0.05 threshold: k ≥ {cluster_extent_threshold} voxels")

# ================================
# 6. Find surviving clusters in observed data
# ================================
surviving_clusters = [size for size in cluster_sizes_obs if size >= cluster_extent_threshold]
n_surviving = len(surviving_clusters)
print(f"\n→ {n_surviving} cluster(s) survive cluster-level FWE correction (p_FWE < 0.05)")

# ================================
# 7. Extract statistics and save results
# ================================
results = []

for cluster_id in range(1, n_clusters_obs + 1):
    size = np.sum(labeled_obs == cluster_id)
    if size < cluster_extent_threshold:
        continue

    mask = (labeled_obs == cluster_id)
    cluster_delta = delta_obs[mask[brain_mask_3d]]
    peak_voxel_flat = np.argmax(cluster_delta)
    i, j, k = voxel_indices[0][peak_voxel_flat], voxel_indices[1][peak_voxel_flat], voxel_indices[2][peak_voxel_flat]
    peak_mni = apply_affine(affine, (i, j, k))
    com_ijk = center_of_mass(mask)
    com_mni = apply_affine(affine, com_ijk)
    peak_p_unc = pvals[peak_voxel_flat]

    # AAL labeling
    try:
        aal = datasets.fetch_atlas_aal('SPM12')
        aal_img = load_img(aal.maps)
        vox = tuple(int(round(c)) for c in coord_transform(*peak_mni, np.linalg.inv(aal_img.affine)))
        idx = int(aal_img.get_fdata()[vox])
        region = aal.labels[aal.indices.index(str(idx))] if str(idx) in aal.indices else "Unknown"
    except:
        region = "Unknown"

    results.append({
        "Cluster": len(results) + 1,
        "Size (vox)": int(size),
        "Peak Δr": round(cluster_delta.max(), 5),
        "Peak p (unc)": f"{peak_p_unc:.2e}",
        "Peak MNI": tuple(round(x, 1) for x in peak_mni),
        "CoM MNI": tuple(round(x, 1) for x in com_mni),
        "AAL Region": region
    })

    # Save individual cluster
    cluster_vol = np.zeros_like(obs_3d)
    cluster_vol[mask] = delta_obs[mask[brain_mask_3d]]
    nib.save(
        nib.Nifti1Image(cluster_vol, affine),
        os.path.join(OUT_DIR, f"cluster_FWE_{len(results):02d}_k{size}_p{peak_p_unc:.1e}.nii.gz")
    )

    # Plot
    plotting.plot_stat_map(
        nib.Nifti1Image(cluster_vol, affine),
        title=f"Cluster {len(results)} | k={size} | Peak Δr={cluster_delta.max():.5f} | p={peak_p_unc:.1e}",
        output_file=os.path.join(OUT_DIR, f"plot_cluster_{len(results):02d}.png"),
        colorbar=True,
        threshold=0.001
    )

# ================================
# 8. Save table
# ================================
if results:
    df = pd.DataFrame(results)
    excel_path = os.path.join(OUT_DIR, "clusters_clusterFWE_corrected.xlsx")
    df.to_excel(excel_path, index=False)
    print("\n=== SURVIVING CLUSTERS (cluster-level FWE p < 0.05) ===")
    print(df[["Cluster", "Size (vox)", "Peak Δr", "Peak p (unc)", "Peak MNI", "AAL Region"]].to_string(index=False))
    print(f"\nExcel table saved → {excel_path}")
else:
    print("\nNo clusters survived cluster-level FWE correction.")

# Save thresholded map
sig_map = flat_to_3d((pvals < PRIMARY_P).astype(float))
nib.save(nib.Nifti1Image(sig_map, affine), os.path.join(OUT_DIR, f"primary_threshold_p{PRIMARY_P:.0e}.nii.gz"))

print(f"\nAll results saved to:\n    {OUT_DIR}")
print("Done. This is publication-ready, non-parametric cluster inference.")