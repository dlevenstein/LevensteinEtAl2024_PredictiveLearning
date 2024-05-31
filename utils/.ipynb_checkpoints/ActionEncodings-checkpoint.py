import torch
from torch import nn
import numpy as np

forwardIDX = 2
LIDX = 0
RIDX = 1

#Note: can remove extra 0s..... but will break old networks. Fix before reruning all networks?
#Or add backwards compatiblity option. Not a huge deal, just extra parameters.

def OneHot(act, obs):
    #Action of -1 means no action
    noActFlag = False
    numActs = 7
    if act[0]<0:
        act = np.ones_like(act)
        noActFlag = True
    
    act = torch.tensor(act, requires_grad=False)
    act = nn.functional.one_hot(act, num_classes=numActs)
    act = torch.unsqueeze(act, dim=0)
    
    if noActFlag:
        act = torch.zeros_like(act)
        
    return act


def addHD(act,obs, suppOffset=False):
    #Unpredictied Observation (e.g. head direction) at time t, 
    #passed in as action
    numSuppObs = 4
    suppObs = 'direction'
    
    if suppOffset:
        #TODO: make this be a function of which pRNN is used
        suppIdx = range(1,len(obs))
    else:
        suppIdx = range(len(obs)-1)
        
    suppObs = np.array([obs[t][suppObs] for t in suppIdx])
    suppObs = torch.tensor(suppObs, dtype=torch.int64, requires_grad=False)
    suppObs = nn.functional.one_hot(suppObs, num_classes=numSuppObs)
    suppObs = torch.unsqueeze(suppObs, dim=0)
    
    act = torch.cat((act,suppObs), dim=2)
    return act


def HDOnly(act, obs):
    act = NoAct(act,obs)
    act = addHD(act,obs)
    return act

def OneHotHD(act, obs):
    act = OneHot(act,obs)
    act = addHD(act,obs)
    return act


def SpeedHD(act, obs):
    act = OneHot(act,obs)
    act[:,:,forwardIDX+1:] = 0
    act[:,:,:forwardIDX] = 0
    act = addHD(act,obs)
    return act

def SpeedNextHD(act, obs):
    act = OneHot(act,obs)
    act[:,:,forwardIDX+1:] = 0
    act[:,:,:forwardIDX] = 0
    act = addHD(act,obs, suppOffset=True)
    return act


def Velocities(act,obs):
    act = OneHot(act,obs)
    act[:,:,forwardIDX+1:] = 0
    act[:,:,LIDX] = act[:,:,LIDX]-act[:,:,RIDX]
    act[:,:,RIDX] = 0
    return act


def NoAct(act,obs):
    act = OneHot(act,obs)
    act = torch.zeros_like(act)
    return act