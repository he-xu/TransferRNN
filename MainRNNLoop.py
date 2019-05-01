#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:21:22 2019

@author: dzenn
"""

# exec(open("MainRNNLoop.py").read())

from DynapseRNN import DynapseRNN
from bias_tuning_tools import BiasTools
import numpy as np
from force import ternarize, update_weight

import pickle

import sys

sys.path.append('/home/dzenn/anaconda3/envs/ctxctl3.7/lib/python3.7/site-packages')



class MainRNNLoop():
    
    def __init__(self, num_inputs = 128, timesteps = 28):        
        
        self.RNNController = DynapseRNN(num_inputs = num_inputs, timesteps = timesteps, debug = True)
        self.data = None
        
    def prepare_dataset(self):
        
        projection_train = open('x_projection_train.pkl', 'rb')
        state_train = open('state_train.pkl', 'rb')

        u_1 = pickle._Unpickler(projection_train)
        u_2 = pickle._Unpickler(state_train)
        u_1.encoding = 'latin1'
        u_2.encoding = 'latin1'
        
        self.projection_train_data = u_1.load()
        self.state_train_data = u_2.load()
        
        
    def start(self):
        pass
        
        
        
#if __name__ == "__main__":
    
bt = BiasTools()
bt.load_biases("best_bias.py")
print("Clearing CAMs...")
bt.clear_cams(1)

print("Initializing the main loop")
MainLoop = MainRNNLoop()

print("Loading the dataset")
MainLoop.prepare_dataset()

print("Loading complete. Starting...")

for image_idx in range(100):
    print("Start recording")
    MainLoop.RNNController.start_recording_spikes()
    print("Showing digit %d" % (image_idx))
    MainLoop.RNNController.present_stimulus(MainLoop.projection_train_data[image_idx], 1/28)
    print("Stopping the recording")
    MainLoop.RNNController.stop_recording_spikes()
    print("Processing recorded evts...")
    rates = MainLoop.RNNController.process_recorded_evts()
    print(np.array(rates))
    
    print("Computing gradients...")
    c_grad, mean_error = MainLoop.RNNController.update_weight(np.array(rates), (MainLoop.state_train_data[image_idx])*50, learning_rate = 0.0008)
    
    MainLoop.RNNController.apply_new_matrix(MainLoop.RNNController.w_ternary, False)
    
    print("C_grad: %g, mean_error %g" % (c_grad, mean_error))
    print("Done")
#    
    

    
    