#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flexible FDR / p-value + cluster analysis
SAFE: no naming conflicts with scipy functions
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image, plotting
from nilearn.image import resample_to_img, load_img
from scipy.ndimage import label as ndimage_label   # ← SAFE: rename the function!
from scipy.ndimage import center_of_mass
from nibabel.affines import apply_affine

# ================================
# USER SETTINGS — CHANGE THESE ONLY
# ================================
PERM_NPZ = "results_wholebrain_irosar/normalized_time/permutation_results/perm_stats_textweighted_opensmile_with_base.npz"
EXAMPLE_NII = "data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz"

THRESHOLD_TYPE   = "unc"      # "fdr" or "unc"
P_VALUE          = 0.001       # e.g. 0.05, 0.001
MIN_CLUSTER_SIZE = 5         # voxels

# ================================
# AUTO-GENERATED NAMES
# ================================
thresh_str = f"{'FDR' if THRESHOLD_TYPE=='fdr' else 'p'}{P_VALUE:.3f}".replace("0.", "")
OUT_DIR = f"results_wholebrain_irosar/normalized_time/permutation_results/significance_maps_{thresh_str}_k{MIN_CLUSTER_SIZE}"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Running: {thresh_str} (k ≥ {MIN_CLUSTER_SIZE}) → {OUT_DIR}")

# ================================
# Load data
# ================================
data = np.load(PERM_NPZ)
delta_obs   = data['delta_obs']
pvals_raw   = data['pvals']
pvals_fdr   = data['pvals_fdr']
reject_fdr  = data['reject']

# ================================
# Recreate exact mask
# ================================
icbm = datasets.fetch_icbm152_2009()
mask_icbm = image.load_img(icbm["mask"])
example_nii = nib.load(EXAMPLE_NII)
resampled_mask = resample_to_img(mask_icbm, example_nii, interpolation="nearest")
brain_mask_3d = resampled_mask.get_fdata() > 0

assert brain_mask_3d.sum() == len(delta_obs), "Mask mismatch!"
affine = example_nii.affine
voxel_indices = np.where(brain_mask_3d)

def flat_to_3d(arr):
    vol = np.zeros(brain_mask_3d.shape, dtype=float)
    vol[voxel_indices] = arr
    return vol

# ================================
# Apply threshold
# ================================
if THRESHOLD_TYPE == "fdr":
    significant = pvals_fdr < P_VALUE
    label_text = f"FDR q < {P_VALUE}"
else:
    significant = pvals_raw < P_VALUE
    label_text = f"p < {P_VALUE} (uncorrected)"

sig_3d = flat_to_3d(significant.astype(float))
delta_3d = flat_to_3d(delta_obs)

nib.save(nib.Nifti1Image(sig_3d, affine),
         os.path.join(OUT_DIR, f"significant_voxels_{thresh_str}.nii.gz"))
nib.save(nib.Nifti1Image(delta_3d, affine),
         os.path.join(OUT_DIR, "delta_r_map.nii.gz"))

print(f"{label_text}: {significant.sum()} significant voxels")

# ================================
# Cluster analysis — NOW SAFE!
# ================================
if significant.sum() == 0:
    print("No significant voxels.")
else:
    structure = np.ones((3,3,3), dtype=bool)
    labeled_array, n_clusters = ndimage_label(sig_3d > 0, structure=structure)  # ← uses renamed function
    print(f"Found {n_clusters} clusters before size filtering")

    clusters = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = (labeled_array == cluster_id)
        size = cluster_mask.sum()
        if size < MIN_CLUSTER_SIZE:
            continue

        values = delta_obs[cluster_mask[brain_mask_3d]]
        peak_val = values.max()
        peak_idx = np.argmax(values)
        i, j, k = voxel_indices[0][peak_idx], voxel_indices[1][peak_idx], voxel_indices[2][peak_idx]
        com_ijk = center_of_mass(cluster_mask)

        peak_mni = apply_affine(affine, (i, j, k))
        com_mni  = apply_affine(affine, com_ijk)

        # AAL label
        try:
            aal = datasets.fetch_atlas_aal('SPM12')
            aal_img = load_img(aal.maps)
            vox = tuple(int(round(c)) for c in image.coord_transform(*peak_mni, np.linalg.inv(aal_img.affine)))
            idx = int(aal_img.get_fdata()[vox])
            region = aal.labels[aal.indices.index(str(idx))] if str(idx) in aal.indices else "Unknown"
        except:
            region = "Unknown"

        clusters.append({
            "Cluster": len(clusters)+1,
            "Size": int(size),
            "Peak Δr": round(peak_val, 4),
            "Mean Δr": round(values.mean(), 4),
            "Peak MNI": tuple(round(x, 1) for x in peak_mni),
            "CoM MNI": tuple(round(x, 1) for x in com_mni),
            "AAL Region": region
        })

        # Save + plot
        clust_vol = np.zeros_like(delta_3d)
        clust_vol[cluster_mask] = delta_3d[cluster_mask]
        nib.save(nib.Nifti1Image(clust_vol, affine),
                 os.path.join(OUT_DIR, f"cluster_{len(clusters):02d}_size{size}.nii.gz"))

        plotting.plot_stat_map(
            nib.Nifti1Image(clust_vol, affine),
            title=f"{thresh_str} | Cluster {len(clusters)} | k={size} | Peak={peak_val:.4f}",
            output_file=os.path.join(OUT_DIR, f"plot_cluster_{len(clusters):02d}.png"),
            colorbar=True,
            threshold=0.001
        )

    if clusters:
        df = pd.DataFrame(clusters)
        df.to_excel(os.path.join(OUT_DIR, f"clusters_{thresh_str}_k{MIN_CLUSTER_SIZE}.xlsx"), index=False)
        print(f"\n{len(clusters)} clusters survived k ≥ {MIN_CLUSTER_SIZE}")
        print(df[["Cluster", "Size", "Peak Δr", "Peak MNI", "AAL Region"]].to_string(index=False))
    else:
        print("No clusters survived the size threshold.")

print(f"\nAll results saved to:\n    {OUT_DIR}")
print("Done!")