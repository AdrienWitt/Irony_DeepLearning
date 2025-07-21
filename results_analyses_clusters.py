import numpy as np
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import resample_to_img
from scipy.ndimage import label
from nilearn.plotting import plot_stat_map
from scipy.ndimage import center_of_mass
from nibabel.affines import apply_affine
import pandas as pd

# 1. Load correlation maps
r_text_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")

# 2. Compute delta_r
delta_r = r_text_audio - np.maximum(r_text, r_audio)
print(f"delta_r shape: {delta_r.shape}")
print(f"Max delta_r: {delta_r.max()}")
print(f"Min delta_r: {delta_r.min()}")


# 3. Load mask and example data
icbm = datasets.fetch_icbm152_2009()
mask_path = icbm['mask']
mask = image.load_img(mask_path)

example_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
resampled_mask = resample_to_img(mask, example_data, interpolation='nearest')
brain_mask = resampled_mask.get_fdata() > 0

# 4. Map delta_r to 3D brain space
delta_r_3d = np.zeros(brain_mask.shape)
delta_r_3d[brain_mask] = delta_r

# 5. Apply threshold (99.9th percentile)
threshold = np.percentile(delta_r, 99.9)
threshold_mask = delta_r_3d > threshold

# 6. Perform clustering with 6-connectivity
structure = np.ones((3, 3, 3))
labeled_array, num_clusters = label(threshold_mask, structure=structure)
print(f"Found {num_clusters} clusters.")

# 7. Compute stats for each cluster
min_size = 5
cluster_stats = []

for cluster_label in range(1, num_clusters + 1):
    cluster_mask = (labeled_array == cluster_label)
    cluster_size = cluster_mask.sum()
    if cluster_size < min_size:
        continue

    cluster_vals = delta_r_3d[cluster_mask]
    cluster_mean = cluster_vals.mean()
    cluster_mass = cluster_vals.sum()
    cluster_max = cluster_vals.max()
    cluster_min = cluster_vals.min()

    # Get peak voxel index in full 3D mask
    peak_idx_flat = np.where(cluster_mask & (delta_r_3d == cluster_max))
    # Note: may be multiple — take first
    peak_voxel = tuple(coord[0] for coord in peak_idx_flat)

    # Compute center of mass in voxel indices
    com_voxel = center_of_mass(cluster_mask)

    # Convert to MNI space using affine
    peak_mni = apply_affine(example_data.affine, peak_voxel)
    com_mni = apply_affine(example_data.affine, com_voxel)

    cluster_stats.append({
        "label": cluster_label,
        "size": cluster_size,
        "mean": cluster_mean,
        "mass": cluster_mass,
        "max": cluster_max,
        "min": cluster_min,
        "peak_voxel": peak_voxel,
        "peak_mni": peak_mni,
        "com_voxel": com_voxel,
        "com_mni": com_mni
    })

print(f"Found {len(cluster_stats)} clusters after size filtering.")

# === 8. Sort by mass ===
cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x['mass'], reverse=True)

# === 9. Print nicely ===
print("\nTop clusters (sorted by mass):")
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    print(f"#{i} | Label: {cluster['label']:>3} | Size: {cluster['size']:>5} vox "
          f"| Mean: {cluster['mean']:.4f} | Mass: {cluster['mass']:.4f} "
          f"| Max: {cluster['max']:.4f} | Min: {cluster['min']:.4f}")
    print(f"    Peak voxel index: {cluster['peak_voxel']}")
    print(f"    Peak MNI coord: {np.round(cluster['peak_mni'], 2)}")
    print(f"    Center-of-mass (voxel): {np.round(cluster['com_voxel'], 2)}")
    print(f"    Center-of-mass (MNI): {np.round(cluster['com_mni'], 2)}\n")


table_data = []
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    table_data.append({
        '#': i,
        'Size (vox)': cluster['size'],
        'Mean': round(cluster['mean'], 4),
        'Mass': round(cluster['mass'], 4),
        'Max': round(cluster['max'], 4),
        'Min': round(cluster['min'], 4),
        'Peak MNI Coord': tuple(round(x, 2) for x in cluster['peak_mni']),
        'CoM MNI': tuple(round(x, 2) for x in cluster['com_mni'])
    })

# Create DataFrame
df = pd.DataFrame(table_data)

# Display the table
print(df.to_string(index=False))

df.to_excel('results_wholebrain_irosar/results_maps/cluster_stats_table.xlsx', index=False)

# 10. Save all clusters in a single NIfTI map with delta_r_3d values
all_clusters_map = np.zeros_like(delta_r_3d)
for cluster in cluster_stats:
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    all_clusters_map[cluster_mask] = delta_r_3d[cluster_mask]
all_clusters_img = nib.Nifti1Image(all_clusters_map, affine=example_data.affine)
nib.save(all_clusters_img, "results_wholebrain_irosar/results_maps/all_clusters_map.nii")
print("\nSaved all clusters in a single NIfTI file with delta_r values: results_wholebrain_irosar/results_maps/all_clusters_map.nii")

# 11. Save and plot top 5 clusters independently
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    cluster_map = np.zeros_like(delta_r_3d)
    cluster_map[cluster_mask] = delta_r_3d[cluster_mask]
    cluster_img = nib.Nifti1Image(cluster_map, affine=example_data.affine)
    nib.save(cluster_img, f"results_wholebrain_irosar/results_maps/top_cluster_{i}.nii")
    # Plot each cluster independently
    plot_stat_map(cluster_img, title=f"Cluster {i} (Label {cluster_label})")

print("\nSaved and plotted top 5 clusters as individual NIfTI files.")