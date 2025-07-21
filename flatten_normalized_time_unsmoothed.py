import os
import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def flatten_nifti(input_path, output_path):
    """Flatten a 3D NIfTI file to 1 x voxels and save as a numpy array."""
    try:
        logging.info(f"Processing {input_path}")
        
        # Verify input file exists
        if not input_path.exists():
            logging.error(f"Input file does not exist: {input_path}")
            return
        
        # Load NIfTI file
        img = nib.load(str(input_path))
        data = img.get_fdata()
        
        # Log data shape
        logging.info(f"Loaded {input_path} with shape {data.shape}")
        
        # Ensure data is 3D (expected 79x95x79)
        if data.ndim != 3:
            logging.error(f"Skipping {input_path}: Expected 3D data, got {data.ndim}D with shape {data.shape}")
            return
        if data.shape != (79, 95, 79):
            logging.warning(f"Unexpected shape for {input_path}: Got {data.shape}, expected (79, 95, 79)")
        
        # Flatten to 1 x voxels (1 x 592,895)
        flattened = data.reshape(1, -1)
        logging.info(f"Flattened shape: {flattened.shape}")
        
        # Verify output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save flattened array
        np.save(output_path, flattened)
        logging.info(f"Saved flattened data to {output_path}")
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")

def process_fmri_files(input_dir, output_dir, num_workers=None):
    """Process all NIfTI files in input_dir in parallel and save to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Collect all NIfTI files and their output paths
    file_pairs = []
    for root, _, files in os.walk(input_dir):
        rel_path = Path(root).relative_to(input_dir)
        out_subdir = output_dir / rel_path
        out_subdir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                input_file = Path(root) / file
                base_name = Path(file).stem
                if base_name.endswith('.nii'):
                    base_name = Path(base_name).stem
                output_file = out_subdir / f"{base_name}.npy"
                file_pairs.append((input_file, output_file))
    
    if not file_pairs:
        logging.warning(f"No NIfTI files found in {input_dir}")
        return
    
    logging.info(f"Found {len(file_pairs)} NIfTI files to process")
    
    # Use number of CPU cores if num_workers is not specified
    logging.info(f"Using {num_workers} worker(s)")
    
    # Process files (serial if num_workers=1, parallel otherwise)

    for input_file, output_file in file_pairs:
        flatten_nifti(input_file, output_file)

    
    logging.info(f"Processed {len(file_pairs)} files")

if __name__ == "__main__":
    try:
        # Example usage
        input_directory = r"data\fmri\normalized_time_unsmoothed"
        output_directory = r"data\fmri\normalized_time_unsmoothed_flatten"
        process_fmri_files(input_directory, output_directory, num_workers=50)
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")