#!/usr/bin/env bash
#SBATCH --array=0-24%25

#COMPUTE CANADA
#SBATCH --account=rrg-bengioy-ad     #Compute canada
#SBATCH --mem-per-cpu=4G   #Note - bigger than 4G moves you to a fat partition on CC, with much fewer jobs. Increase cpus-per-task instead
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/dlevenst/sbatch_out/masked1_hyperparm_panel.%A.%a.out  #compute canada
#SBATCH --job-name=masked1_hyperparm_panel
source ~/projects/def-tyrell-ab/dlevenst/PredictiveReplay/load_venv.sh

#MILA
# #SBATCH --partition=long-cpu
# #SBATCH --mem=12GB
# #SBATCH --time=48:00:00
# #SBATCH --cpus-per-task=1
# #SBATCH --output=/network/scratch/d/daniel.levenstein/sbatch_out/masked1_hyperparm_panel.%A.%a.out
# #SBATCH --job-name=masked1_hyperparm_panel
# source ~/PredictiveReplay/load_venv.sh

SEED=${1:-8}
ITERATE=$SLURM_ARRAY_TASK_ID

TRAINPARMS=($(python hyperparm_panel.py --s=$SEED --i=$ITERATE | tr -d '[],'))

echo $TRAINPARMS

python trainNet.py --savefolder='hyperparm_panel_masked1/' --pRNNtype='thRNN_1win' --actenc='SpeedHD' --seqdur=${TRAINPARMS[1]} --lr=${TRAINPARMS[0]} --bptttrunc=${TRAINPARMS[2]} --hiddensize=${TRAINPARMS[4]} --weight_decay=${TRAINPARMS[3]} --ntimescale=${TRAINPARMS[5]} --dropout=0.15 --noisestd=0.03 --bias_lr=${TRAINPARMS[9]} --sparsity=0.5 --trainBias --numepochs=60 --namext=$ITERATE -s=$SEED --no-saveTrainData