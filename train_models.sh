#!/bin/bash

################################################################################
#
# Submit file for a batch job on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# See the manpages for salloc, srun, sbatch, squeue, scontrol, and scancel
# for more information or read the Slurm docs online: https://slurm.schedmd.com
#
################################################################################

#SBATCH --mail-type=ALL
#SBATCH --mail-user=schneideral@msoe.edu

# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes.  Another option is the 'batch' partition.
#SBATCH --partition=dgx

# The number of nodes to request
#SBATCH --nodes=1

# The number of GPUs to request
#SBATCH --gpus=4

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=2

# The error file to write to
#SBATCH --error='sbatcherrorfile.out'

# Kill the job if it takes longer than the specified time
# format: <days>-<hours>:<minutes>
#SBATCH --time=1-0:0
# Path to container

container="/data/containers/msoe-tensorflow-24.05-tf2-py3.sif"

# Execute singularity container on node.
singularity exec --nv -B /data:/data -B /home:/home ${container} python3 train_models.py
