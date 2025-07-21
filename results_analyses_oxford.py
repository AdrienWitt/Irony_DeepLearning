import numpy as np
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import resample_to_img
from scipy.ndimage import label
from nilearn.plotting import plot_stat_map

# 1. Load correlation maps
r_text_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")

# 2. Compute delta_r
delta_r = r_text_audio - np.maximum(r_text, r_audio)
print(f"delta_r shape: {delta_r.shape}")
print(f"Max delta_r: {delta_r.max()}")

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

# 5. Apply threshold (90th percentile)
threshold = np.percentile(delta_r, 99.9)
threshold_mask = delta_r_3d > threshold

# 6. Initial clustering with 6-connectivity
structure = np.ones((3, 3, 3))

labeled_array, num_clusters = label(threshold_mask, structure=structure)
print(f"Found {num_clusters} initial clusters.")

# # 7. Load and resample AAL atlas
# atlas = datasets.fetch_atlas_aal()
# atlas_img = nib.load(atlas['maps'])
# resampled_atlas = resample_to_img(atlas_img, example_data, interpolation='nearest')
# atlas_data = resampled_atlas.get_fdata()

# Load cortical and subcortical Harvard-Oxford atlases
cort_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
sub_atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
cort_atlas_img = cort_atlas['maps']  # Nifti1Image
sub_atlas_img = sub_atlas['maps']  # Nifti1Image

# Resample both to match example_data
resampled_cort_atlas = resample_to_img(cort_atlas_img, example_data, interpolation='nearest')
resampled_sub_atlas = resample_to_img(sub_atlas_img, example_data, interpolation='nearest')

# Combine cortical and subcortical data (optional: merge or process separately)
atlas_data = resampled_cort_atlas.get_fdata() + resampled_sub_atlas.get_fdata()


# 8. Split clusters by atlas regions
min_size = 1
new_labeled_array = np.zeros_like(labeled_array)
new_cluster_label = 1
cluster_stats = []

for cluster_label in range(1, num_clusters + 1):
    cluster_voxels = (labeled_array == cluster_label)
    if cluster_voxels.sum() < min_size:
        continue
    # Get unique atlas regions in this cluster
    atlas_labels = np.unique(atlas_data[cluster_voxels])
    for alabel in atlas_labels:
        if alabel == 0:  # Skip background
            continue
        # Create a new cluster for each atlas region
        new_cluster_mask = cluster_voxels & (atlas_data == alabel)
        if new_cluster_mask.sum() < min_size:
            continue
        new_labeled_array[new_cluster_mask] = new_cluster_label
        # Compute stats for the new cluster
        cluster_size = new_cluster_mask.sum()
        cluster_mean = delta_r_3d[new_cluster_mask].mean()
        cluster_mass = delta_r_3d[new_cluster_mask].sum()
        cluster_max =  delta_r_3d[new_cluster_mask].max()
        cluster_min =  delta_r_3d[new_cluster_mask].min()

        # Get AAL region name
        cluster_stats.append({
            "label": new_cluster_label,
            "size": cluster_size,
            "mean": cluster_mean,
            "mass": cluster_mass,
            "max" : cluster_max,
            "min" : cluster_min,
        })
        new_cluster_label += 1

num_clusters = new_cluster_label - 1
labeled_array = new_labeled_array
print(f"Found {num_clusters} clusters after atlas-based splitting.")

# 10. Sort clusters by mass (descending)
cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x['mass'], reverse=True)

# 11. Print top clusters
print("\nTop clusters (sorted by mass):")
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    print(f"#{i} | Label: {cluster['label']:>3} | Size: {cluster['size']:>5} vox | Mean: {cluster['mean']:.4f} | Mass: {cluster['mass']:.4f}")

# 12. Save all clusters in a single NIfTI map with delta_r_3d values
all_clusters_map = np.zeros_like(delta_r_3d)
for cluster in cluster_stats:
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    all_clusters_map[cluster_mask] = delta_r_3d[cluster_mask]
all_clusters_img = nib.Nifti1Image(all_clusters_map, affine=example_data.affine)
nib.save(all_clusters_img, "results_wholebrain_irosar/results_maps/all_clusters_map.nii")
print("\nSaved all clusters in a single NIfTI file with delta_r values: results_wholebrain/results_maps/all_clusters_map_oxford.nii")

# 13. Save and plot top 10 clusters independently
for i, cluster in enumerate(cluster_stats_sorted[:5], 1):
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    cluster_map = np.zeros_like(delta_r_3d)
    cluster_map[cluster_mask] = delta_r_3d[cluster_mask]
    cluster_img = nib.Nifti1Image(cluster_map, affine=example_data.affine)
    nib.save(cluster_img, f"results_wholebrain_irosar/results_maps/top_cluster_{i}_oxford.nii")
    # Plot each cluster independently
    plot_stat_map(cluster_img, title=f"Cluster {i} (Label {cluster_label}")

print("\nSaved and plotted top 5 clusters as individual NIfTI files.")








