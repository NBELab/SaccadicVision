#!/usr/bin/env bash
  
## Resource allocation settings

#SBATCH --ntasks=1                  # Number of tasks to run (1 task)
#SBATCH --cpus-per-task=1            # Number of CPU cores per task (1 core)
#SBATCH --mem=70g                    # Memory allocation (70GB)
#SBATCH --nodelist=gpu7              # Specific node to use (gpu7)
#SBATCH --gres=gpu:a100:1            # GPU allocation (1 A100 GPU)
#SBATCH --qos=gpu                    # Job quality of service (using GPU queue)

## Job details

#SBATCH --job-name=train               # Name of the job (can be customized, here it's 'FOO')
#SBATCH --output=outputs/%x-%j.out    # File to save standard output (%x = job name, %j = job ID)
#SBATCH --error=outputs/%x-%j.err     # File to save error output (%x = job name, %j = job ID)

## Email notifications

#SBATCH --mail-type=ALL              # Send email for all job events (start, end, fail)
#SBATCH --mail-user=YahiaShowgan@gmail.com # Email address to send notifications

## Load required modules and activate environment

module purge                         # Clear any pre-loaded modules to avoid conflicts
module load anaconda3                # Load Anaconda module
source /home/shyahia/myenv/bin/activate         # Activate your custom environment (path to the environment)

## Run the Python script

python ../main.py                       # Run the main Python script

## Post-job actions

hostname                             # Print the hostname (useful for debugging)
sleep 1                              # Pause for 1 second before ending the script
