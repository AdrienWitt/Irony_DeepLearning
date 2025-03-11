#!/bin/sh
#SBATCH --job-name CV_IRONY          # this is a parameter to help you sort your job when listing it
#SBATCH --error cv_results/step1_err      # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output cv_results/step1_out      # optional. By default the error and output files are merged
#SBATCH --ntasks 50                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1
#SBATCH --mem=96000             # number of cpus for each task. One by default
#SBATCH --time 24:00:00                  # maximum run time.
#SBATCH --partition=Main
#SBATCH --nodes=1                 # Use 1 node
#SBATCH --nodelist=cisa-calc3.unige.ch  # Specify node explicitly

source envs/py39/bin/activate
python3 CV.py --step 1 --fixed_alpha 100.0 --use_text --use_audio --use_base_features --num_jobs 50 --step2_use_best_base   