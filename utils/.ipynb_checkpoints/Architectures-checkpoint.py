#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:17:33 2022

@author: dl2820
"""

""" Preditive net modules below """

from torch import nn
import torch
import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import norm
from utils.thetaRNN import thetaRNNLayer, RNNCell, LayerNormRNNCell, AdaptingLayerNormRNNCell, AdaptingRNNCell

from utils.pytorchInits import CANN_


class pRNN(nn.Module):
    """
    A general predictive RNN framework that takes observations and actions, and
    returns predicted observations, as well as the actual observations to train
    Observations are inputs that are predicted. Actions are inputs that are
    relevant to prediction but not predicted.

    predOffset: the output at time t s matched to obs t+predOffset (defulat: 1)
    actOffset:  the action input at time t is the action took at t-actOffset (default:0)
    inMask:     boolean list, length corresponding to the prediction cycle period. (default: [True])
    outMask, actMask: list, same length as inMask (default: None)

    All inputs should be tensors of shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
    """
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=RNNCell,  dropp=0, trunc=50, k=0, f=0.5,
                 predOffset=1, inMask=[True], outMask=None, cyclePeriod=1,
                 actOffset=0, actMask=None, neuralTimescale=2,
                continuousTheta=False):
        super(pRNN, self).__init__()

        #pRNN architecture parameters
        self.predOffset = predOffset
        self.actOffset = actOffset
        self.actpad = nn.ConstantPad1d((0,0,self.actOffset,0),0)
        self.inMask = inMask
        if outMask is None:
            outMask = [True for i in inMask]
        if actMask is None:
            actMask = [True for i in inMask]
        self.outMask = outMask
        self.actMask = actMask
        
        self.droplayer = nn.Dropout(p=dropp)
        #self.droplayer_act = nn.Dropout(p=dropp_act)
        
        #Sparsity via layernorm subtraction
        mu = norm.ppf(f)
        musig = [mu,1]

        #TODO: add cellparams input to pass through
        #Consider putting the sigmoid outside this layer...
        input_size = obs_size + act_size
        self.rnn = thetaRNNLayer(cell, trunc, input_size, hidden_size, musig, 
                                 defaultTheta=k, continuousTheta=continuousTheta)
        self.outlayer = nn.Sequential(
            nn.Linear(hidden_size, obs_size, bias=False),
            nn.Sigmoid()
            )

        self.W_in = self.rnn.cell.weight_ih
        self.W = self.rnn.cell.weight_hh
        self.W_out = self.outlayer[0].weight
        self.bias = self.rnn.cell.bias
        
        self.neuralTimescale = neuralTimescale

        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(1-1/self.neuralTimescale))


    def forward(self, obs, act, noise_t=torch.tensor([]), state=torch.tensor([]), theta=None):
        x_t, obs_target, outmask = self.restructure_inputs(obs,act)
        #x_t = self.droplayer(x_t) # (should it count action???) dropout with action
        h_t,_ = self.rnn(x_t,internal=noise_t, state=state, theta=theta)
        allout = self.outlayer(h_t)

        #Apply the mask to the output
        y_t = torch.zeros_like(allout)
        y_t[:,outmask,:] = allout[:,outmask,:] #The predicted outputs.
        return y_t, h_t, obs_target

   #TODO: combine forward and internal?
    def internal(self, noise_t, state=torch.tensor([])):
        h_t,_ = self.rnn(internal=noise_t, state=state, theta=0)
        y_t = self.outlayer(h_t)
        return y_t, h_t

    def restructure_inputs(self, obs, act):
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """

        #Apply the action and prediction offsets
        act = self.actpad(act)
        obs_target = obs[:,self.predOffset:,:]

        #Make everything the same size
        minsize = min(obs.size(1),act.size(1),obs_target.size(1))
        obs, act = obs[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]

        #Apply the masks (this is ugly.)
        actmask = np.resize(np.array(self.actMask),minsize)
        outmask = np.resize(np.array(self.outMask),minsize)
        obsmask = np.resize(np.array(self.inMask),minsize)

        obs_out = torch.zeros_like(obs, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)

        obs_out[:,obsmask,:] = obs[:,obsmask,:]
        act_out[:,actmask,:] = act[:,actmask,:]
        obs_target_out[:,outmask,:] = obs_target[:,outmask,:]
        
        obs_out = self.droplayer(obs_out) #dropout without action
        #act_out = self.droplayer_act(act_out) #dropout without action

        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask



class pRNN_th(pRNN):    
    def __init__(self, obs_size, act_size, k, hidden_size=500,
                 cell=RNNCell,  dropp=0, trunc=50, f=0.5,
                 predOffset=0, inMask=[True], outMask=None,
                 actOffset=0, actMask=None, neuralTimescale=2,
                continuousTheta=False, actionTheta=False):
        super(pRNN_th, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                 cell=cell,  dropp=dropp, trunc=trunc, k=k, f=0.5,
                 predOffset=predOffset, inMask=inMask, outMask=outMask,
                 actOffset=actOffset, actMask=actMask,
                                      neuralTimescale=neuralTimescale,
                                     continuousTheta=continuousTheta)
        
        self.k = k
        self.actionTheta = actionTheta
        
        
    def restructure_inputs(self, obs, act):
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size/Theta Size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """
        #Apply the action and prediction offsets
        act = self.actpad(act)
        obs_target = obs[:,self.predOffset:,:]
        
        #Apply the theta prediction for target observation
        theta_idx = np.flip(toeplitz(np.arange(self.k+1),
                                     np.arange(obs_target.size(1))),0)
        theta_idx = theta_idx[:,self.k:,]
        obs_target = obs_target[:,theta_idx.copy()]
        obs_target = torch.squeeze(obs_target,0)
        
        if self.actionTheta == 'hold':
            act = act.expand(self.k+1,-1,-1)
            obs = nn.functional.pad(input=obs, pad=(0,0,0,0,0,self.k), 
                                    mode='constant', value=0)


        elif self.actionTheta is True:
            theta_idx = np.flip(toeplitz(np.arange(self.k+1),
                                         np.arange(act.size(1))),0)
            theta_idx = theta_idx[:,self.k:,]
            act = act[:,theta_idx.copy()]
            act = torch.squeeze(act,0)
            obs = nn.functional.pad(input=obs, pad=(0,0,0,0,0,self.k), 
                                    mode='constant', value=0)
            
        
        #Make everything the same size
        minsize = min(obs.size(1),act.size(1),obs_target.size(1))
        obs, act = obs[:,:minsize,:], act[:,:minsize,:]
        obs_target = obs_target[:,:minsize,:]
        
        #No masks for theta net
        obs_out = torch.zeros_like(obs, requires_grad=False)
        act_out = torch.zeros_like(act, requires_grad=False)
        obs_target_out = torch.zeros_like(obs_target, requires_grad=False)
        outmask = True
        
        obs_out[:,:,:] = obs[:,:,:]
        act_out[:,:,:] = act[:,:,:]
        obs_target_out[:,:,:] = obs_target[:,:,:]

        obs = self.droplayer(obs) #dropout without action
        
        #Concatenate the obs/act into a single input
        x_t = torch.cat((obs_out,act_out), 2)
        return x_t, obs_target_out, outmask
        
        
        
        
        

        
        
        
        
        

class RNN2L(nn.Module):
    """
    A vanilla RNN architecture that takes observations and actions and returns
    observations (i.e. at the next timestep)
    """
    def __init__(self,obs_size, act_size, hidden_size=500, encsize=300,
                 nonlinearity='relu'):
        super(RNN2L, self).__init__()

        input_size = obs_size + act_size

        #Functions/layers
        self.inlayer = nn.Linear(input_size, encsize, bias=False)
        self.enc_rnn = nn.RNN(encsize, encsize, bias=False, batch_first=True,
                          nonlinearity = nonlinearity)
        #TODO: Check act
        self.FFlayer = nn.Linear(encsize,hidden_size, bias=False)
        self.p_rnn = nn.RNN(hidden_size, hidden_size, bias=False, batch_first=True,
                          nonlinearity = nonlinearity)


        self.W_internal2 = self.p_rnn.weight_ih_l0
        self.W_internal1 = self.enc_rnn.weight_ih_l0
        self.W_in = self.inlayer.weight
        self.W_FF = self.FFlayer.weight
        self.W = self.p_rnn.weight_hh_l0
        self.W_enc = self.enc_rnn.weight_hh_l0


        #TODO Clean this
        with torch.no_grad():
            self.W_internal2 = torch.nn.Parameter(torch.eye(hidden_size,
                                                           requires_grad=False))
            self.W_internal1 = torch.nn.Parameter(torch.eye(encsize,
                                                           requires_grad=False))
            self.W.add_(torch.eye(hidden_size).div_(2))
            self.W_enc.add_(torch.eye(encsize).div_(2))
            #self.W_out = torch.nn.Parameter(self.W_in.t())
        #TODO: Initalization as optional param


    def forward(self, obs, act):
        x_t = self.restructure_inputs(obs,act)
        x_t = self.inlayer(x_t)
        e_t,_ = self.enc_rnn(x_t)
        x_t = self.FFlayer(e_t)

        obs_next = torch.zeros_like(x_t)
        obs_next[:,1::2,:] = x_t[:,1::2,:] #The predicted outputs.

        obs_in = torch.zeros_like(x_t)
        obs_in[:,0::2,:] = x_t[:,0::2,:] #The FF input, every other timestep.

        h_t,_ = self.p_rnn(obs_in)
        #Input to last layer is obs_next.
        y_t = torch.zeros_like(h_t)
        y_t[:,1::2,:] = h_t[:,1::2,:] #The predicted outputs.

        return y_t, h_t, obs_next

    def internal(self, noise_t):
        h_t,_ = self.p_rnn(noise_t)
        #y_t = self.outlayer(h_t)
        return h_t, h_t


    def restructure_inputs(self, obs, act):
        """
        Join obs and act into a single input tensor shape (N,L,H)
        N: Batch size
        L: timesamps
        H: input_size
        obs should be one timestep longer than act, for the [t+1] observation
        after the last action
        """
        act = torch.cat((act,torch.zeros(act.size(0),1,act.size(2))), dim=1)
        obs = torch.cat((obs,torch.zeros(obs.size(0),1,obs.size(2))), dim=1)

        obs = obs[:,:-1,:]
        x_t = torch.cat((obs,act), 2)
        return x_t


    
    

class vRNN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell):
        super(vRNN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=None, actMask=None)

class thRNN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell):
        super(thRNN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True], actMask=None)

class vRNN_0win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_0win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class vRNN_1win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_1win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,True], outMask=[True,True], actMask=None)

class vRNN_2win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_2win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class vRNN_3win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_3win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class vRNN_4win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_4win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class vRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)

class vRNN_1win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_1win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])

class vRNN_2win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_2win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=[True,False,False])

class vRNN_3win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_3win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True], 
                          actMask=[True,False,False,False])

class vRNN_4win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_4win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=[True,False,False,False,False])

class vRNN_5win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(vRNN_5win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=[True,False,False,False,False,False])



        
        
class thRNN_0win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_0win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_1win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_2win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_3win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_4win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_5win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win_noLN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_6win_noLN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
        
        
        
        
        
        


class thRNN_0win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_0win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)

class thRNN_1win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_1win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=None)

class thRNN_2win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_2win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=None)

class thRNN_3win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_3win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=None)

class thRNN_4win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_4win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=None)

class thRNN_5win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_5win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_6win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_6win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_7win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_7win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_8win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_8win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          f=f,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_9win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_9win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
class thRNN_10win(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thRNN_10win, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False,False,False,False,False,False], 
                                         outMask=[True,True,True,True,True,True,True,True,True,True,True],
                          actMask=None)
        
        

class thRNN_1win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thRNN_1win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])

class thRNN_2win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thRNN_2win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False], outMask=[True,True,True],
                          actMask=[True,False,False])

class thRNN_3win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thRNN_3win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False], outMask=[True,True,True,True],
                          actMask=[True,False,False,False])

class thRNN_4win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thRNN_4win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False], outMask=[True,True,True,True,True],
                          actMask=[True,False,False,False,False])

class thRNN_5win_mask(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thRNN_5win_mask, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False,False,False,False,False], outMask=[True,True,True,True,True,True],
                          actMask=[True,False,False,False,False,False])






class AutoencoderFF(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=None):
        super(AutoencoderFF, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
class AutoencoderRec(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=None):
        super(AutoencoderRec, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderPred(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=None):
        super(AutoencoderPred, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderFFPred(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=RNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=None):
        super(AutoencoderFFPred, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
        
        
class AutoencoderFF_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(AutoencoderFF_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()
        
class AutoencoderRec_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(AutoencoderRec_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderPred_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(AutoencoderPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                                                 f=f,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        
class AutoencoderFFPred_LN(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0, f=0.5):
        super(AutoencoderFFPred_LN, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                                                   f=f,
                          predOffset=1, actOffset=0,
                          inMask=[True], outMask=[True], actMask=None)
        self.W.requires_grad_(False)
        self.W.zero_()

class AutoencoderMaskedO(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(AutoencoderMaskedO, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True],
                          actMask=[True,True])
        
class AutoencoderMaskedOA(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(AutoencoderMaskedOA, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[False,True],
                          actMask=[True,False])

class AutoencoderMaskedO_noout(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(AutoencoderMaskedO_noout, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,True])
        
class AutoencoderMaskedOA_noout(pRNN):
    def __init__(self, obs_size, act_size, hidden_size=500,
                      cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(AutoencoderMaskedOA_noout, self).__init__(obs_size, act_size, hidden_size=hidden_size,
                          cell=cell, trunc=trunc, neuralTimescale=neuralTimescale, dropp=dropp,
                          predOffset=0, actOffset=0,
                          inMask=[True,False], outMask=[True,True],
                          actMask=[True,False])


        

        
        
class thcycRNN_3win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thcycRNN_3win, self).__init__(obs_size, act_size,  k=3, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      )
        
class thcycRNN_4win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thcycRNN_4win, self).__init__(obs_size, act_size,  k=4, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      )
        
class thcycRNN_5win(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0):
        super(thcycRNN_5win, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True)
        
        
        
class thcycRNN_5win_holdc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_holdc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_fullc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_firstc, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_hold, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_full, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=LayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_first, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=False, actionTheta=False)
        
        
        
        
class thcycRNN_5win_holdc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_holdc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True, actionTheta='hold')
        
class thcycRNN_5win_fullc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_fullc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True, actionTheta=True)
        
class thcycRNN_5win_firstc_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_firstc_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=True, actionTheta=False)
        
        
class thcycRNN_5win_hold_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_hold_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=False, actionTheta='hold')
        
class thcycRNN_5win_full_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_full_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=False, actionTheta=True)
        
class thcycRNN_5win_first_adapt(pRNN_th):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 cell=AdaptingLayerNormRNNCell, trunc=50, neuralTimescale=2, dropp = 0,
                f=0.5):
        super(thcycRNN_5win_first_adapt, self).__init__(obs_size, act_size,  k=5, 
                                       hidden_size=hidden_size,
                                      cell=cell, trunc=trunc, 
                                       neuralTimescale=neuralTimescale, dropp=dropp,
                                        f=f,
                                       predOffset=0, actOffset=0,
                                      continuousTheta=False, actionTheta=False)
        
        
        
        
        

class vRNN_LayerNorm(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_LayerNorm, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)


class thRNN_LayerNorm(thRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(thRNN_LayerNorm, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)


class vRNN_LayerNormAdapt(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_LayerNormAdapt, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=AdaptingLayerNormRNNCell)



class vRNN_CANN(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_CANN, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=LayerNormRNNCell)

        #TODO Clean this
        size = [20,20,20]
        Nmaps = 1
        self.locations = CANN_(self.W, size, Nmaps, selfconnect=False)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(0.5))



class vRNN_adaptCANN(vRNN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_adaptCANN, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias,
                                             cell=AdaptingLayerNormRNNCell)

        #TODO Clean this
        size = [15,15,15]
        Nmaps = 1
        self.locations = CANN_(self.W, size, Nmaps, selfconnect=False)
        with torch.no_grad():
            self.W.add_(torch.eye(hidden_size).mul_(0.5))




class vRNN_CANN_FFonly(vRNN_CANN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_CANN_FFonly, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias)

        rootk = np.sqrt(1/hidden_size)
        with torch.no_grad():
            self.W.add_(torch.rand(hidden_size, hidden_size)*0.5*rootk)
        self.W.requires_grad=False


class vRNN_adptCANN_FFonly(vRNN_adaptCANN):
    def __init__(self,obs_size, act_size, hidden_size=500,
                 nonlinearity='relu', bias=False):
        super(vRNN_adptCANN_FFonly, self).__init__(obs_size, act_size,
                                             hidden_size=hidden_size,
                                             nonlinearity=nonlinearity,
                                             bias=bias)

        rootk = np.sqrt(1/hidden_size)
        with torch.no_grad():
            self.W.add_(torch.rand(hidden_size, hidden_size)*0.2*rootk)
        self.W.requires_grad=False
