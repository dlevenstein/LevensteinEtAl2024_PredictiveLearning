#!/usr/bin/env bash
#SBATCH --array=0-21%22

#COMPUTE CANADA
#SBATCH --account=rrg-bengioy-ad     #Compute canada
#SBATCH --mem-per-cpu=4G   #Note - bigger than 4G moves you to a fat partition on CC, with much fewer jobs. Increase cpus-per-task instead
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/dlevenst/sbatch_out/Autoencoder_sparse_panel.%A.%a.out  #compute canada
#SBATCH --job-name=Autoencoder_sparse_panel
source ~/projects/def-tyrell-ab/dlevenst/PredictiveReplay/load_venv.sh

#MILA
# #SBATCH --partition=long-cpu
# #SBATCH --mem=12GB
# #SBATCH --time=48:00:00
# #SBATCH --cpus-per-task=1
# #SBATCH --output=/network/scratch/d/daniel.levenstein/sbatch_out/Autoencoder_sparse_panel.%A.%a.out
# #SBATCH --job-name=Autoencoder_sparse_panel
# source ~/PredictiveReplay/load_venv.sh


lamda_arr=('AutoencoderPred_LN' 'AutoencoderRec_LN' 'AutoencoderFF_LN'  'AutoencoderFFPred_LN' 'AutoencoderMaskedO' 'AutoencoderMaskedOA' 'AutoencoderMaskedO_noout' 'AutoencoderMaskedOA_noout' 'AutoencoderPred_LN' 'AutoencoderPred_LN' 'AutoencoderPred_LN' 'AutoencoderPred_LN' 'AutoencoderPred_LN' 'AutoencoderRec_LN' 'AutoencoderMaskedO' 'AutoencoderMaskedOA' 'AutoencoderMaskedO_noout' 'AutoencoderMaskedOA_noout' 'AutoencoderMaskedO' 'AutoencoderMaskedO' 'AutoencoderMaskedO' 'AutoencoderMaskedO') 

act_arr=('Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'Onehot' 'OnehotHD' 'SpeedHD' 'Velocities' 'SpeedNextHD' 'NoAct' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'SpeedHD' 'OnehotHD' 'Velocities' 'SpeedNextHD' 'NoAct' )

#'AutoencoderMaskedO' 'AutoencoderMaskedOA' 'AutoencoderMaskedO_noout' 'AutoencoderMaskedOA_noout')

python trainNet.py --savefolder='Autoencoder_sparse_panel/' --env='MiniGrid-LRoom-18x18-v0' --lr=2e-3 --sparsity=0.5 --noisestd=0.03 --dropout=0.15 --numepochs=80 --ntimescale=2  --hiddensize=500 --seqdur=600 --bias_lr=0.1 --trainBias --pRNNtype=${lamda_arr[$SLURM_ARRAY_TASK_ID]} --actenc=${act_arr[$SLURM_ARRAY_TASK_ID]} --namext=${act_arr[$SLURM_ARRAY_TASK_ID]} -s=${1:-8} --no-saveTrainData
