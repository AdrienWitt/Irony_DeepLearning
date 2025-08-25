import numpy as np
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import resample_to_img, smooth_img
from scipy.ndimage import label
from nilearn.plotting import plot_stat_map
from scipy.ndimage import center_of_mass
from nibabel.affines import apply_affine
import pandas as pd
from nilearn.image import load_img, coord_transform


# 1. Load correlation maps
r_text_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")

# 2. Compute delta_r
delta_r = r_text_audio - np.maximum(r_text, r_audio)

# 3. Function to print stats
def print_stats(name, data):
    print(f"{name}: Min = {data.min():.4f}, Max = {data.max():.4f}, M = {data.mean():.4f}, SD = {data.std():.4f}")

# 4. Print stats for each map
print_stats("Text-Audio", r_text_audio)
print_stats("Text", r_text)
print_stats("Audio", r_audio)
print_stats("Delta r", delta_r)

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
    peak_mni = apply_affine(resampled_mask.affine, peak_voxel)
    com_mni = apply_affine(resampled_mask.affine, com_voxel)

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
cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x['mean'], reverse=True)

# === 9. Print nicely ===
print("\nTop clusters (sorted by mass):")
for i, cluster in enumerate(cluster_stats_sorted[:15], 1):
    print(f"#{i} | Label: {cluster['label']:>3} | Size: {cluster['size']:>5} vox "
          f"| Mean: {cluster['mean']:.4f} | Mass: {cluster['mass']:.4f} "
          f"| Max: {cluster['max']:.4f} | Min: {cluster['min']:.4f}")
    print(f"    Peak voxel index: {cluster['peak_voxel']}")
    print(f"    Peak MNI coord: {np.round(cluster['peak_mni'], 2)}")
    print(f"    Center-of-mass (voxel): {np.round(cluster['com_voxel'], 2)}")
    print(f"    Center-of-mass (MNI): {np.round(cluster['com_mni'], 2)}\n")


def get_brain_region(mni_coords, atlas='destrieux'):
    atlas_data = datasets.fetch_atlas_destrieux_2009(legacy_format=True)
    atlas_img = load_img(atlas_data['maps'])
    atlas_labels = atlas_data['labels']
    
    voxel_coords = tuple(round(x) for x in coord_transform(*mni_coords, np.linalg.inv(atlas_img.affine)))
    region_index = int(atlas_img.get_fdata()[voxel_coords])
    
    return atlas_labels[region_index][1] if region_index < len(atlas_labels) else 'Unknown'


def get_brain_region_aal(mni_coords, atlas='aal'):
    # Fetch the AAL atlas
    atlas_data = datasets.fetch_atlas_aal(version='SPM12')
    atlas_img = load_img(atlas_data['maps'])
    atlas_labels = atlas_data['labels']
    atlas_indices = atlas_data['indices']  # AAL provides indices for mapping

    # Transform MNI coordinates to voxel coordinates in atlas space
    voxel_coords = coord_transform(*mni_coords, np.linalg.inv(atlas_img.affine))
    voxel_coords = tuple(int(round(x)) for x in voxel_coords)  # Round to nearest integer

    # Get the region index from the atlas
    region_index = int(atlas_img.get_fdata()[voxel_coords])
    
    return atlas_labels[atlas_indices.index(str(region_index))] if region_index in [int(i) for i in atlas_indices] else "Unknown"
    

def get_brain_region_ho(mni_coords, atlas='cortical', threshold=0):
    # Validate atlas type
    if atlas not in ['cortical', 'subcortical']:
        raise ValueError("Atlas must be 'cortical' or 'subcortical'.")

    # Fetch the Harvard-Oxford atlas
    if atlas == 'cortical':
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
    else:
        atlas_data = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-1mm')

    atlas_img = load_img(atlas_data['maps'])
    atlas_labels = atlas_data['labels']

    # Transform MNI coordinates to voxel coordinates in atlas space
    voxel_coords = coord_transform(*mni_coords, np.linalg.inv(atlas_img.affine))
    voxel_coords = tuple(int(round(x)) for x in voxel_coords)  # Round to nearest integer

    # Get the region index from the atlas
    region_index = int(atlas_img.get_fdata()[voxel_coords])

    # Return the label if the index is valid, else "Unknown"
    if region_index > 0 and region_index < len(atlas_labels):
        return atlas_labels[region_index]
    else:
        return "Unknown"


table_data = []
for i, cluster in enumerate(cluster_stats_sorted, 1):
    region = get_brain_region_aal(cluster['peak_mni'])
    table_data.append({
        '#': i,
        'Size (vox)': cluster['size'],
        'Mean': round(cluster['mean'], 4),
        'Mass': round(cluster['mass'], 4),
        'Max': round(cluster['max'], 4),
        'Min': round(cluster['min'], 4),
        'Peak MNI Coord': tuple(round(x, 2) for x in cluster['peak_mni']),
        'CoM MNI': tuple(round(x, 2) for x in cluster['com_mni']),
        'Region': region
    })

df = pd.DataFrame(table_data)
print(df.to_string(index=False))


df.to_excel('results_wholebrain_irosar/results_maps/cluster_stats_table_999.xlsx', index=False)

# 10. Save all clusters in a single NIfTI map with delta_r_3d values
all_clusters_map = np.zeros_like(delta_r_3d)
for cluster in cluster_stats:
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    all_clusters_map[cluster_mask] = delta_r_3d[cluster_mask]
all_clusters_img = nib.Nifti1Image(all_clusters_map, affine=example_data.affine)
nib.save(all_clusters_img, "results_wholebrain_irosar/results_maps/all_clusters_map_999.nii")
print("\nSaved all clusters in a single NIfTI file with delta_r values: results_wholebrain_irosar/results_maps/all_clusters_map.nii")

# # 10b. Save top 10 clusters in a single NIfTI map with delta_r_3d values
# top_clusters_map = np.zeros_like(delta_r_3d)
# for cluster in cluster_stats_sorted[:15]:
#     cluster_label = cluster['label']
#     cluster_mask = (labeled_array == cluster_label)
#     top_clusters_map[cluster_mask] = delta_r_3d[cluster_mask]
# top_clusters_img = nib.Nifti1Image(top_clusters_map, affine=example_data.affine)
# nib.save(top_clusters_img, "results_wholebrain_irosar/results_maps/top_10_clusters_map.nii")
# print("\nSaved top 10 clusters in a single NIfTI file with delta_r values: results_wholebrain_irosar/results_maps/top_10_clusters_map.nii")

# 11. Save and plot top clusters independently
for i, cluster in enumerate(cluster_stats_sorted, 1):
    cluster_label = cluster['label']
    cluster_mask = (labeled_array == cluster_label)
    cluster_map = np.zeros_like(delta_r_3d)
    cluster_map[cluster_mask] = delta_r_3d[cluster_mask]
    cluster_img = nib.Nifti1Image(cluster_map, affine=example_data.affine)
    nib.save(cluster_img, f"results_wholebrain_irosar/results_maps/top_cluster_{i}_999.nii")
    # Plot each cluster independently
    plot_stat_map(cluster_img, title=f"Cluster {i} (Label {cluster_label})")

print("\nSaved and plotted top 10 clusters as individual NIfTI files.")




