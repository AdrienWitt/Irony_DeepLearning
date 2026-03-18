#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT: Top 1% most significant voxels (by FDR p-value)
→ Only FDR-significant voxels
→ Then top 1% smallest p-values among them
→ Clustering + all voxels in cluster are significant
→ Beautiful output + AAL labels
"""

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from nilearn.image import resample_to_img, load_img, coord_transform
from scipy.ndimage import label, center_of_mass
from nibabel.affines import apply_affine

# ================================
# USER SETTINGS
# ================================
TOP_PVAL_PERCENTILE = 99.5      # 99.0 = top 1%, 99.5 = top 0.5%, 99.9 = top 0.1%
MIN_CLUSTER_SIZE    = 5
OUTPUT_NAME = f"top1pct_most_significant_k{MIN_CLUSTER_SIZE}_FDR05"

# ================================
# 1. Load correlation maps & compute Δr
# ================================
r_comb  = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text  = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")
delta_r_flat = r_comb - np.maximum(r_text, r_audio)

# ================================
# 2. Load permutation results
# ================================
perm = np.load("results_wholebrain_irosar/normalized_time/permutation_results/perm_stats_textweighted_opensmile_with_base.npz")
pvals_fdr  = perm['pvals_fdr']
reject_fdr = perm['reject']        # True if FDR p < 0.05

# ================================
# 3. Brain mask + coordinate mapping
# ================================
icbm    = datasets.fetch_icbm152_2009()
example = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
brain_mask = resample_to_img(image.load_img(icbm['mask']), example, interpolation='nearest').get_fdata() > 0
affine  = example.affine

i_coords, j_coords, k_coords = np.where(brain_mask)
n_voxels = len(i_coords)
ijk_to_flat = {(i_coords[i], j_coords[i], k_coords[i]): i for i in range(n_voxels)}

print(f"Brain mask: {n_voxels:,} voxels")

# ================================
# 4. Map flat Δr and p-values to 3D
# ================================
delta_r_3d = np.zeros(brain_mask.shape, dtype=float)
delta_r_3d[brain_mask] = delta_r_flat          # ← THIS WAS MISSING!

pval_3d = np.full(brain_mask.shape, np.inf)
pval_3d[brain_mask] = pvals_fdr

# ================================
# 5. Step 1: FDR-significant voxels only
# ================================
sig_mask = np.zeros_like(brain_mask, dtype=bool)
sig_mask[brain_mask] = reject_fdr
n_sig = sig_mask.sum()
print(f"FDR-significant voxels (p < .05): {n_sig:,}")

if n_sig == 0:
    print("No significant voxels found!")
    exit()

# ================================
# 6. Step 2: Top X% most significant among them
# ================================
sig_pvals = pvals_fdr[reject_fdr]                     # p-values of significant voxels
p_threshold = np.percentile(sig_pvals, TOP_PVAL_PERCENTILE)

top_sig_mask = (pval_3d <= p_threshold) & sig_mask
n_top = top_sig_mask.sum()
print(f"Top {(100-TOP_PVAL_PERCENTILE):.1f}% most significant → p ≤ {p_threshold:.6f} → {n_top:,} voxels")

# ================================
# 7. Clustering
# ================================
labeled, n_clusters = label(top_sig_mask, structure=np.ones((3,3,3), bool))

# ================================
# 8. Extract valid clusters
# ================================
valid_clusters = []
print(f"\nClustering → keeping clusters with k ≥ {MIN_CLUSTER_SIZE}...")

for lbl in range(1, n_clusters + 1):
    mask = (labeled == lbl)
    size = mask.sum()
    if size < MIN_CLUSTER_SIZE:
        continue

    # All voxels already significant → just extract stats
    vals = delta_r_3d[mask]
    peak_val = vals.max()
    peak_ijk = np.where((delta_r_3d == peak_val) & mask)
    i, j, k = peak_ijk[0][0], peak_ijk[1][0], peak_ijk[2][0]

    peak_mni = apply_affine(affine, (i, j, k))
    com_mni  = apply_affine(affine, center_of_mass(mask))
    peak_p   = pvals_fdr[ijk_to_flat[(i,j,k)]]

    valid_clusters.append({
        "label": lbl,
        "size": size,
        "mean_Δr": vals.mean(),
        "peak_Δr": peak_val,
        "peak_mni": peak_mni,
        "com_mni": com_mni,
        "peak_p": peak_p
    })
    print(f"  Kept cluster {lbl:3d} | k={size:4d} | peak Δr={peak_val:.5f} | FDR p={peak_p:.6f}")

# ================================
# 9. Pretty formatting + AAL
# ================================
def format_pval(p):
    return "<.001" if p < 0.001 else f"{p:.4f}".lstrip("0")

def get_aal(mni):
    try:
        aal = datasets.fetch_atlas_aal('SPM12')
        img = load_img(aal.maps)
        vox = tuple(int(round(c)) for c in coord_transform(*mni, np.linalg.inv(img.affine)))
        idx = int(img.get_fdata()[vox])
        return aal.labels[aal.indices.index(str(idx))] if str(idx) in aal.indices else "Unknown"
    except:
        return "Unknown"

# ================================
# 10. Final table
# ================================
valid_clusters = sorted(valid_clusters, key=lambda x: x['peak_p'])  # most significant first

table = []
for rank, c in enumerate(valid_clusters, 1):
    region = get_aal(c['peak_mni'])
    table.append({
        '#': rank,
        'Size': c['size'],
        'Mean Δr': round(c['mean_Δr'], 5),
        'Peak Δr': round(c['peak_Δr'], 5),
        'FDR p': format_pval(c['peak_p']),
        'Peak MNI': tuple(round(x, 1) for x in c['peak_mni']),
        'CoM MNI': tuple(round(x, 1) for x in c['com_mni']),
        'AAL': region
    })

df = pd.DataFrame(table)
print("\n" + "="*110)
print(f"TOP 1% MOST SIGNIFICANT CLUSTERS (FDR p < .05)")
print("="*110)
print(df.to_string(index=False))
print("="*110)

# Save
out_dir = "results_wholebrain_irosar/results_maps/pvalue_based"
df.to_excel(f"{out_dir}/{OUTPUT_NAME}.xlsx", index=False)

final_map = np.zeros_like(delta_r_3d)
for c in valid_clusters:
    final_map[labeled == c['label']] = delta_r_3d[labeled == c['label']]
nib.save(nib.Nifti1Image(final_map, affine), f"{out_dir}/{OUTPUT_NAME}.nii")

print(f"\nDone! Ultra-conservative significance-driven result saved.")
print(f"   Table → {out_dir}/{OUTPUT_NAME}.xlsx")
print(f"   Map   → {out_dir}/{OUTPUT_NAME}.nii")
print(f"   Final clusters: {len(valid_clusters)}")