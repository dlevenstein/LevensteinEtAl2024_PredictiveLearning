#!/usr/bin/env bash
#SBATCH --array=0-27%28

#COMPUTE CANADA
#SBATCH --account=rrg-bengioy-ad     #Compute canada
#SBATCH --mem-per-cpu=4G   #Note - bigger than 4G moves you to a fat partition on CC, with much fewer jobs. Increase cpus-per-task instead
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/dlevenst/sbatch_out/maskedk_panel.%A.%a.out  #compute canada
#SBATCH --job-name=maskedk_panel
source ~/projects/def-tyrell-ab/dlevenst/PredictiveReplay/load_venv.sh

#MILA
# #SBATCH --partition=long-cpu
# #SBATCH --mem=12GB
# #SBATCH --time=48:00:00
# #SBATCH --cpus-per-task=1
# #SBATCH --output=/network/scratch/d/daniel.levenstein/sbatch_out/maskedk_panel.%A.%a.out
# #SBATCH --job-name=maskedk_panel
# source ~/PredictiveReplay/load_venv.sh

lamda_arr=('thRNN_0win' 'thRNN_1win' 'thRNN_2win' 'thRNN_3win' 'thRNN_4win' 'thRNN_5win' 'thRNN_6win' 'thRNN_0win' 'thRNN_1win' 'thRNN_2win' 'thRNN_3win' 'thRNN_4win' 'thRNN_5win' 'thRNN_6win' 'thRNN_0win' 'thRNN_1win' 'thRNN_2win' 'thRNN_3win' 'thRNN_4win' 'thRNN_5win' 'thRNN_6win' 'thRNN_0win' 'thRNN_1win' 'thRNN_2win' 'thRNN_3win' 'thRNN_4win' 'thRNN_5win' 'thRNN_6win')

#lamda_arr=('thRNN_0win_noLN' 'thRNN_1win_noLN' 'thRNN_2win_noLN' 'thRNN_3win_noLN' 'thRNN_4win_noLN' 'thRNN_5win_noLN' 'thRNN_6win_noLN' 'thRNN_0win_noLN' 'thRNN_1win_noLN' 'thRNN_2win_noLN' 'thRNN_3win_noLN' 'thRNN_4win_noLN' 'thRNN_5win_noLN' 'thRNN_6win_noLN' 'thRNN_0win_noLN' 'thRNN_1win_noLN' 'thRNN_2win_noLN' 'thRNN_3win_noLN' 'thRNN_4win_noLN' 'thRNN_5win_noLN' 'thRNN_6win_noLN' 'thRNN_0win_noLN' 'thRNN_1win_noLN' 'thRNN_2win_noLN' 'thRNN_3win_noLN' 'thRNN_4win_noLN' 'thRNN_5win_noLN' 'thRNN_6win_noLN')

act_arr=('SpeedHD' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'OnehotHD' 'OnehotHD' 'OnehotHD' 'OnehotHD' 'OnehotHD' 'OnehotHD' 'OnehotHD'
'Velocities' 'Velocities' 'Velocities' 'Velocities' 'Velocities' 'Velocities' 'Velocities')

python trainNet.py --savefolder='maskedk_panel/' --actenc=${act_arr[$SLURM_ARRAY_TASK_ID]} --env='MiniGrid-LRoom-18x18-v0' --lr=2e-3 --sparsity=0.5 --noisestd=0.03 --dropout=0.15 --numepochs=80 --ntimescale=2 --hiddensize=500 --seqdur=600 --bias_lr=0.1 --trainBias --pRNNtype=${lamda_arr[$SLURM_ARRAY_TASK_ID]} --namext=${act_arr[$SLURM_ARRAY_TASK_ID]} -s=${1:-8} --no-saveTrainData
