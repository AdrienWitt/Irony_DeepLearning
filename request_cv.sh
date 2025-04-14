#!/bin/sh
#SBATCH --job-name CV_IRONY          # this is a parameter to help you sort your job when listing it
#SBATCH --error text_err      # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output text_out      # optional. By default the error and output files are merged
#SBATCH --ntasks 50                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1
#SBATCH --mem=96000             # number of cpus for each task. One by default
#SBATCH --time 24:00:00                  # maximum run time.
#SBATCH --partition=Main
#SBATCH --nodes=1                 # Use 1 node
#SBATCH --nodelist=cisa-calc3.unige.ch  # Specify node explicitly

source envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export JOBLIB_TEMP_FOLDER=/tmp

python3 CV.py --use_text --use_base_features --num_jobs 50 --step 2 --pca_thresholds 0.50 0.60 0.70 --fixed_alpha 0.1