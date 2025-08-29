import numpy as np

# Load the data
nc = np.load("results_wholebrain_irosar/normalized_time/mean_noise_ceiling.npy")
r_text_audio = np.load("results_wholebrain_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")

# Ensure arrays are valid
print("Shape of nc:", nc.shape)
print("Shape of r_text_audio:", r_text_audio.shape)

# Find the maximum correlation in r_text_audio
max_r_text_audio = np.max(r_text_audio)
max_r_index = np.argmax(r_text_audio)  # Index of the maximum value

# Compare with noise ceiling
if nc.shape == r_text_audio.shape:
    nc_at_max_r = nc[max_r_index]  # Noise ceiling at the same location as max r_text_audio
    print(f"Maximum r_text_audio: {max_r_text_audio:.4f}")
    print(f"Noise ceiling at max r_text_audio location: {nc_at_max_r:.4f}")
    print(f"Ratio (r_text_audio / nc): {max_r_text_audio / nc_at_max_r:.4f}")
else:
    print("Shape mismatch between nc and r_text_audio. Comparing max only.")
    print(f"Maximum r_text_audio: {max_r_text_audio:.4f}")
    print(f"Mean noise ceiling: {np.mean(nc):.4f}")

# Optional: Check how many r_text_audio values are close to nc
threshold = 0.9  # Example: Consider values within 90% of nc
close_to_ceiling = np.sum(r_text_audio / nc > threshold) if nc.shape == r_text_audio.shape else 0
print(f"Number of r_text_audio values > {threshold*100}% of noise ceiling: {close_to_ceiling}")