#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:07:04 2022

@author: dl2820
"""





#%%
from utils.predictiveNet import PredictiveNet
from utils.agent import RandomActionAgent
from utils.env import make_env
from utils.figures import TrainingFigure
from utils.figures import SpontTrajectoryFigure
from analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis
import argparse

#TODO: get rid of these dependencies
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters

parser.add_argument("--env", default='MiniGrid-LRoom-18x18-v0',
                    help="name of the environment to train on (Default: MiniGrid-LRoom-18x18-v0)")

# parser.add_argument("--agent", default='RandomActionAgent',
#                     help="name of the environment to train on (Default: RandomActionAgent)")

parser.add_argument("--pRNNtype", default='thRNN_2win',
                    help="which pRNN (Default: thRNN_2win)")

parser.add_argument("--savefolder", default='',
                    help="Where to save the net? (foldername/)")

parser.add_argument("--loadfolder", default='',
                    help="Where to load the net? (foldername/)")

parser.add_argument("--numepochs", default=30, type=int,
                     help="how many training epochs? (Default: 40)")

parser.add_argument("--seqdur", default=500, type=int,
                     help="how long is each behavioral sequence? (Default: 1000")

parser.add_argument("--numtrials", default=1000, type=int,
                     help="many trials in an eqpoch? (Default: 1000")

parser.add_argument("--hiddensize", default=500, type=int,
                     help="how many hidden units? (Default: 300")

parser.add_argument("-c", "--contin", action="store_true",
                     help="Continue previous training?")

parser.add_argument("--load_env", default=-1, type=int,
                     help="Load Environment for continued Training. Specify unique env id")

parser.add_argument("-s", "--seed", default=8, type=int,
                     help="Random Seed? (Default: 8)")

parser.add_argument("--lr", default=3e-3, type=float,    #former default:2e-4 (not relative)
                     help="Learning Rate? (Relative to init sqrt(1/k) for each layer) (Default: 1e-3)")

parser.add_argument("--weight_decay", default=3e-3, type=float, #former default:6e-7 (not relative)
                     help="Weight Decay? (Relative to learning rate) (Default: 0)")

parser.add_argument("--bptttrunc", default=1e8, type=int,
                     help="BPTT Truncation window? (Default: 0)")

parser.add_argument("--ntimescale", default=2, type=float,
                     help="Neural timescale (Default: 2 timesteps)")

parser.add_argument("--dropout", default=0.15, type=float,
                     help="Dropout probability (Default: 0)")

parser.add_argument("--noisemean", default=0, type=float,
                     help="Mean offset for internal noise (Default: 0)")

parser.add_argument("--noisestd", default=0.03, type=float,
                     help="Std of internal noise (Default: 0)")

parser.add_argument("-f", "--sparsity", default=0.5, type=float,
                     help="Activation sparsity (via layer norm, irrelevant for non-LN networks) (Default: 0.5)")

parser.add_argument('--trainBias', action='store_true', default=False)

parser.add_argument("--bias_lr", default=1, type=float,    #former default:2e-4 (not relative)
                     help="Bias Learning Rate? (Relative to learning rate) (Default: 1)")

parser.add_argument('--identityInit', action='store_true', default=False)

parser.add_argument("--namext", default='',
                     help="Extension to the savename?")

parser.add_argument("--actenc", default='OnehotHD',
                     help="Action encoding, options: OnehotHD (default),SpeedHD, Onehot, Velocities")

parser.add_argument('--saveTrainData', action='store_true', default=True)
parser.add_argument('--no-saveTrainData', dest='saveTrainData', action='store_false')

args = parser.parse_args()



savename = args.pRNNtype + '-' + args.namext + '-s' + str(args.seed)
figfolder = 'nets/'+args.savefolder+'/trainfigs/'+savename
analysisfolder = 'nets/'+args.savefolder+'/analysis/'+savename




#%%
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.contin:
    predictiveNet = PredictiveNet.loadNet(args.loadfolder+savename)
    if args.env == '':
        env = predictiveNet.loadEnvironment(args.load_env)
        predictiveNet.addEnvironment(env)
    else:
        env = make_env(args.env)
        predictiveNet.addEnvironment(env)
    agentname = 'RandomActionAgent'
    action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
    agent = RandomActionAgent(env.action_space,action_probability)
else:
    env = make_env(args.env)
    agentname = 'RandomActionAgent'
    action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
    agent = RandomActionAgent(env.action_space,action_probability)
    predictiveNet = PredictiveNet(env, 
                                  hidden_size=args.hiddensize,
                                  pRNNtype=args.pRNNtype,
                                  actionEncoding=args.actenc,
                                  learningRate = args.lr,
                                  bptttrunc = args.bptttrunc,
                                  weight_decay = args.weight_decay,
                                  neuralTimescale = args.ntimescale,
                                  dropp=args.dropout,
                                  trainNoiseMeanStd= (args.noisemean,args.noisestd),
                                  f = args.sparsity,
                                 trainBias = args.trainBias,
                                  bias_lr = args.bias_lr,
                                 identityInit = args.identityInit)
    predictiveNet.seed = args.seed
    predictiveNet.trainArgs = args
    predictiveNet.plotSampleTrajectory(env,agent,
                                       savename=savename+'exTrajectory_untrained',
                                       savefolder=figfolder)
    predictiveNet.savefolder = args.savefolder
    predictiveNet.savename = savename



#%% Training Epoch
#Consider these as "trainingparameters" class/dictionary
numepochs = args.numepochs
sequence_duration = args.seqdur
num_trials = args.numtrials

predictiveNet.trainingCompleted = False
if predictiveNet.numTrainingTrials == -1:
    #Calculate initial spatial metrics etc
    print('Training Baseline')
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=sequence_duration,
                            num_trials=1)
    print('Calculting Spatial Representation...')
    place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                  trainDecoder=True,saveTrainingData=True,
                                                  bitsec= False,
                                                  calculatesRSA = True, sleepstd=0.03)
    predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)
    print('Calculting Decoding Performance...')
    predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                savename=savename, savefolder=figfolder,
                                                saveTrainingData=True)
    #predictiveNet.plotDelayDist(env, agent, decoder)

#TODO: Put in time counter here and ETA...
#TODO: take this out later. for backwards compatibility
if hasattr(predictiveNet, 'numTrainingEpochs') is False:
    predictiveNet.numTrainingEpochs = int(predictiveNet.numTrainingTrials/num_trials)
    
while predictiveNet.numTrainingEpochs<numepochs:
    print(f'Training Epoch {predictiveNet.numTrainingEpochs}')
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=sequence_duration,
                            num_trials=num_trials)
    print('Calculting Spatial Representation...')
    place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                 trainDecoder=True, trainHDDecoder = True,
                                                 saveTrainingData=True, bitsec= False,
                                                 calculatesRSA = True, sleepstd=0.03)
    print('Calculting Decoding Performance...')
    predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                savename=savename, savefolder=figfolder,
                                                saveTrainingData=True)
    predictiveNet.plotLearningCurve(savename=savename,savefolder=figfolder,
                                    incDecode=True)
    #predictiveNet.plotSampleTrajectory(env,agent,savename=savename,savefolder=figfolder)
    predictiveNet.plotTuningCurvePanel(savename=savename,savefolder=figfolder)

    OTA = OfflineTrajectoryAnalysis(predictiveNet,actionAgent=None, noisestd=0.03,
                                   decoder=decoder, calculateViewSimilarity=True,
                                   compareWake=True)
    OTA.SpontTrajectoryFigure(savename+'_noise',figfolder)
    #OTA.saveAnalysis(savename+'_noise',analysisfolder)
    predictiveNet.addTrainingData('replay_alpha_noise',OTA.diffusionFit['alpha'])
    predictiveNet.addTrainingData('replay_int_noise',OTA.diffusionFit['intercept'])
    predictiveNet.addTrainingData('replay_view_noise',OTA.ViewSimilarity['meanstd_sleep'][0][0])
    predictiveNet.addTrainingData('replay_coherence_noise',OTA.spatialCoherence_SLEEP['meanCoherence'])
    predictiveNet.addTrainingData('replay_extent_noise',OTA.spatialCoherence_SLEEP['meanExtent'])
    predictiveNet.addTrainingData('replay_coherence_wake',OTA.spatialCoherence_WAKE['meanCoherence'])
    predictiveNet.addTrainingData('replay_extent_wake',OTA.spatialCoherence_WAKE['meanExtent'])
    predictiveNet.addTrainingData('replay_view_wake',OTA.ViewSimilarity['meanstd_wake'][0][0])
    
    OTA = OfflineTrajectoryAnalysis(predictiveNet,actionAgent=agent, noisestd=0.03,
                                   decoder=decoder, calculateViewSimilarity=True,
                                   compareWake=True)
    OTA.SpontTrajectoryFigure(savename+'_query',figfolder)
    #OTA.saveAnalysis(savename+'_query',analysisfolder)
    predictiveNet.addTrainingData('replay_alpha_query',OTA.diffusionFit['alpha'])
    predictiveNet.addTrainingData('replay_int_query',OTA.diffusionFit['intercept'])
    predictiveNet.addTrainingData('replay_view_query',OTA.ViewSimilarity['meanstd_sleep'][0][0])
    predictiveNet.addTrainingData('replay_coherence_query',OTA.spatialCoherence_SLEEP['meanCoherence'])
    predictiveNet.addTrainingData('replay_extent_query',OTA.spatialCoherence_SLEEP['meanExtent'])
    
    OTA = OfflineTrajectoryAnalysis(predictiveNet, noisestd=0.03,
                               decoder=decoder, calculateViewSimilarity=True,
                               withAdapt=True,b_adapt = 0.3, tau_adapt=8,
                                   compareWake=True)
    OTA.SpontTrajectoryFigure(savename+'_adapt',figfolder)
    #OTA.saveAnalysis(savename+'_adapt',analysisfolder)
    predictiveNet.addTrainingData('replay_alpha_adapt',OTA.diffusionFit['alpha'])
    predictiveNet.addTrainingData('replay_int_adapt',OTA.diffusionFit['intercept'])
    predictiveNet.addTrainingData('replay_view_adapt',OTA.ViewSimilarity['meanstd_sleep'][0][0])
    predictiveNet.addTrainingData('replay_coherence_adapt',OTA.spatialCoherence_SLEEP['meanCoherence'])
    predictiveNet.addTrainingData('replay_extent_adapt',OTA.spatialCoherence_SLEEP['meanExtent'])

    
    plt.show()
    plt.close('all')
    predictiveNet.saveNet(args.savefolder+savename)

predictiveNet.trainingCompleted = True
TrainingFigure(predictiveNet,savename=savename,savefolder=figfolder)

#If the user doesn't want to save all that training data, delete it except the last one
if args.saveTrainData is False:
    predictiveNet.TrainingSaver = predictiveNet.TrainingSaver.drop(predictiveNet.TrainingSaver.index[:-1])
    predictiveNet.saveNet(args.savefolder+savename)