import numpy as np
from scipy.ndimage import label
from nilearn import datasets, image
from nilearn.image import resample_to_img
import nibabel as nib
from nilearn.plotting import plot_stat_map

# 1. Load correlation maps
r_text_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")

# 2. Compute delta_r
delta_r = r_text_audio - np.maximum(r_text, r_audio)
print(f"delta_r shape: {delta_r.shape}")
print(f"delta_r stats: min={delta_r.min():.4f}, max={delta_r.max():.4f}, mean={delta_r.mean():.4f}, std={delta_r.std():.4f}")

# 3. Load mask and example data
icbm = datasets.fetch_icbm152_2009()
mask = image.load_img(icbm['mask'])
example_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
resampled_mask = resample_to_img(mask, example_data, interpolation='nearest')
brain_mask = resampled_mask.get_fdata() > 0
print(f"Number of voxels in brain mask: {brain_mask.sum()}")

# 4. Map delta_r to 3D brain space
delta_r_3d = np.zeros(brain_mask.shape)
delta_r_3d[brain_mask] = delta_r

# 5. Compute voxel-wise p-values using permutations
n_permutations = 1000  # Adjust based on computational resources
perm_p_values = np.zeros_like(delta_r)
for i in range(n_permutations):
    perm_delta_r = np.random.choice([-1, 1], size=delta_r.shape) * delta_r  # Sign flipping
    perm_p_values += (np.abs(perm_delta_r) >= np.abs(delta_r)) / n_permutations
print(f"Permutation p-values: min={perm_p_values.min():.4f}, max={perm_p_values.max():.4f}")

# 6. Apply lenient voxel-wise threshold (p < 0.05, uncorrected)
threshold_p = 0.05
threshold_mask_3d = np.zeros_like(delta_r_3d)
threshold_mask_3d[brain_mask] = perm_p_values < threshold_p
print(f"Voxels with p < {threshold_p} (uncorrected): {threshold_mask_3d.sum()}")

# Alternative: Use percentile threshold if permutation yields too few voxels
if threshold_mask_3d.sum() < 100:  # Arbitrary cutoff to check if enough voxels survive
    percentile = 99.5  # Try 99.5th or 99th percentile
    threshold_value = np.percentile(delta_r, percentile)
    threshold_mask_3d = delta_r_3d > threshold_value
    print(f"Using {percentile}th percentile threshold ({threshold_value:.4f}): {threshold_mask_3d.sum()} voxels")

# 7. Cluster significant voxels
structure = np.ones((3, 3, 3))
labeled_array, num_clusters = label(threshold_mask_3d, structure=structure)
print(f"Found {num_clusters} clusters before size filtering.")

# 8. Compute stats for each cluster
min_size = 3  # Small to capture clusters with small delta_r
cluster_stats = []

for cluster_label in range(1, num_clusters + 1):
    cluster_mask = (labeled_array == cluster_label)
    cluster_size = cluster_mask.sum()
    if cluster_size < min_size:
        continue
    cluster_mean = delta_r_3d[cluster_mask].mean()
    cluster_mass = delta_r_3d[cluster_mask].sum()
    cluster_max = delta_r_3d[cluster_mask].max()
    cluster_min = delta_r_3d[cluster_mask].min()
    
    cluster_stats.append({
        "label": cluster_label,
        "size": cluster_size,
        "mean": cluster_mean,
        "mass": cluster_mass,
        "max": cluster_max,
        "min": cluster_min
    })

print(f"Found {len(cluster_stats)} clusters after size filtering (min_size={min_size}).")

# 9. Sort clusters by mass
cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x['mass'], reverse=True)

# 10. Print top clusters
print("\nTop clusters (sorted by mass):")
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    print(f"#{i} | Label: {cluster['label']:>3} | Size: {cluster['size']:>5} vox | Mean: {cluster['mean']:.4f} | Mass: {cluster['Thermostat
    ['mass']:.4f} | Max: {cluster['max']:.4f}")

# 11. Save all clusters in a single NIfTI map
all_clusters_map = np.zeros_like(delta_r_3d)
for cluster in cluster_stats:
    cluster_mask = (labeled_array == cluster['label'])
    all_clusters_map[cluster_mask] = delta_r_3d[cluster_mask]
all_clusters_img = nib.Nifti1Image(all_clusters_map, affine=example_data.affine)
nib.save(all_clusters_img, "results_wholebrain_irosar/results_maps/permutation_clusters_map.nii")
print("\nSaved all clusters in a single NIfTI file: results_wholebrain_irosar/results_maps/permutation_clusters_map.nii")

# 12. Save and plot top 5 clusters
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    cluster_map = np.zeros_like(delta_r_3d)
    cluster_map[cluster_mask] = delta_r_3d[cluster_mask]
    cluster_img = nib.Nifti1Image(cluster_map, affine=example_data.affine)
    nib.save(cluster_img, f"results_wholebrain_irosar/results_maps/permutation_top_cluster_{i}.nii")
    plot_stat_map(cluster_img, title=f"Cluster {i} (Label {cluster_label})")

print("\nSaved and plotted top 5 clusters as individual NIfTI files.")