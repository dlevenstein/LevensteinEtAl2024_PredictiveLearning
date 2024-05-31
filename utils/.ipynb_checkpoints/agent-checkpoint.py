#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:05:03 2021

@author: dl2820
"""

from numpy.random import choice
import numpy as np

def randActionSequence(tsteps,action_space,action_probability):
    
    action_space = np.arange(action_space.n) #convert gym to np
    action_sequence = choice(action_space, size=(tsteps,), p=action_probability)
    
    return action_sequence
    


class RandomActionAgent:
    def __init__(self, action_space, default_action_probability=None):
        
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = np.ones_lke(self.action_space)/np.size(self.action_space)
        
        
    def generateActionSequence(self, tsteps, action_probability=None):
        if action_probability is None:
            action_probability = self.default_action_probability
        action_sequence = randActionSequence(tsteps,
                                             self.action_space, action_probability)
        return action_sequence
    
    
    def getObservations(self, env, tsteps, reset=True, includeRender=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps)

        render = False
        if reset is False:
            raise ValueError('Reset=False not implemented yet...')
            
        obs = [None for t in range(tsteps+1)]
        obs[0] = env.reset()
        state = {'agent_pos': np.resize(env.agent_pos,(1,2)), 
                 'agent_dir': env.agent_dir
                }
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.render(mode=None)
            
        for aa in range(tsteps):
            obs[aa+1], reward, done, _ = env.step(act[aa])
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.agent_pos,(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.agent_dir)
            if includeRender:
                render[aa+1] = env.render(mode=None)

        return obs, act, state, render
    
    
class RandomHDAgent:
    def __init__(self, action_space, default_action_probability=None, constantAction=-1):
        
        self.action_space = action_space
        self.default_action_probability = default_action_probability
        if default_action_probability is None:
            self.default_action_probability = np.ones_lke(self.action_space)/np.size(self.action_space)
        self.constantAction = constantAction
        
        
    def generateActionSequence(self, tsteps, action_probability=None):
        if action_probability is None:
            action_probability = self.default_action_probability
        action_sequence = randActionSequence(tsteps,
                                             self.action_space, action_probability)
        return action_sequence
    
    
    def getObservations(self, env, tsteps, reset=True, includeRender=False):   
        """
        Get a sequence of observations. act[t] is the action after observing
        obs[t], obs[t+1] is the resulting observation. obs will be 1 entry 
        longer than act
        """
        act = self.generateActionSequence(tsteps)

        render = False
        if reset is False:
            raise ValueError('Reset=False not implemented yet...')
            
        obs = [None for t in range(tsteps+1)]
        obs[0] = env.reset()
        state = {'agent_pos': np.resize(env.agent_pos,(1,2)), 
                 'agent_dir': env.agent_dir
                }
        if includeRender:
            render = [None for t in range(tsteps+1)]
            render[0] = env.render(mode=None)
            
        for aa in range(tsteps):
            obs[aa+1], reward, done, _ = env.step(act[aa])
            state['agent_pos'] = np.append(state['agent_pos'],
                                           np.resize(env.agent_pos,(1,2)),axis=0)
            state['agent_dir'] = np.append(state['agent_dir'],
                                           env.agent_dir)
            if includeRender:
                render[aa+1] = env.render(mode=None)
                
        act = np.ones_like(act) * self.constantAction

        return obs, act, state, render
    