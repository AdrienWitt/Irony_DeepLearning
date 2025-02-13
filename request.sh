#!/bin/sh
#SBATCH --job-name CV_IRONY          # this is a parameter to help you sort your job when listing it
#SBATCH --error cv_results/CV_IRONY_err2      # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output cv_results/CV_IRONY_out2      # optional. By default the error and output files are merged
#SBATCH --ntasks 50                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1
#SBATCH --mem=64000             # number of cpus for each task. One by default
#SBATCH --time 6:00:00                  # maximum run time.
#SBATCH --partition=shared-cpu

module load GCCcore/10.3.0
module load Python/3.9.5     
source envs/py39/bin/activate

python3 CV.py --mode text_audio --num_jobs 50 --alpha_values 600000 700000 1000000 1500000 2000000 2500000 