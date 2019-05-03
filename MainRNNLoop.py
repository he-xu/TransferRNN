#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:21:22 2019

@author: dzenn
"""

#exec(open("MainRNNLoop.py").read())

from DynapseRNN import DynapseRNN
from bias_tuning_tools import BiasTools
import numpy as np
import datetime

import pickle

import sys

sys.path.append('/home/dzenn/anaconda3/envs/ctxctl3.7/lib/python3.7/site-packages')



class MainRNNLoop():
    
    def __init__(self, num_inputs = 128, timesteps = 28):        
        
        self.RNNController = DynapseRNN(num_inputs = num_inputs, timesteps = timesteps, multiplex_factor = 1, debug = True)
        self.recorded_error = []
        
    def prepare_dataset(self):
        
        projection_train = open('x_projection_train.pkl', 'rb')
        state_train = open('state_train.pkl', 'rb')
        mnist = open('mnist.pkl','rb')

        u_1 = pickle._Unpickler(projection_train)
        u_2 = pickle._Unpickler(state_train)
        u_3 = pickle._Unpickler(mnist)
        u_1.encoding = 'latin1'
        u_2.encoding = 'latin1'
        u_3.encoding = 'latin1'
        
        self.projection_train_data = u_1.load()
        self.state_train_data = u_2.load()
#        self.mnist_train, self.mnist_test, self.mnist_validate = u_3.load()
        
    def export_error(self, filename = "RNNError.dat"):
        """
            Export all recorded events
        """
#        f_out = open(str(datetime.datetime.now()) + " " + filename, "wb")        
        f_out = open(filename, "w")
        for err in self.recorded_error:
            f_out.write(str(err) + "\n")
        f_out.close()
        
        
    def start(self):
        pass
        
def run_loop(num_digits):

    for image_idx in range(num_digits):
        print("Start recording")
        MainLoop.RNNController.start_recording_spikes()
        print("Showing digit %d" % (image_idx))
        MainLoop.RNNController.present_stimulus(MainLoop.projection_train_data[1], 2/6)
        print("Stopping the recording")
        MainLoop.RNNController.stop_recording_spikes()
        print("Processing recorded evts...")
        rates = MainLoop.RNNController.process_recorded_evts()
        print(np.array(rates))
        
        print("Computing gradients...")
        c_grad, mean_error = MainLoop.RNNController.update_weight(np.array(rates), (MainLoop.state_train_data[1])*400, learning_rate = 0.01)
        
        MainLoop.recorded_error.append(mean_error)
        
        MainLoop.RNNController.apply_new_matrix(MainLoop.RNNController.w_ternary, False)
        
        print("C_grad: %g, mean_error %g" % (c_grad, mean_error))
        print("Done")        
        
#if __name__ == "__main__":
    
bt = BiasTools()
bt.load_biases("reasonable_rnn_biases.py")
bt.copy_biases(4,5)
bt.copy_biases(4,12)
print("Clearing CAMs...")
bt.clear_cams(1)
bt.clear_cams(3)

print("Initializing the main loop")
MainLoop = MainRNNLoop()

print("Loading the dataset")
MainLoop.prepare_dataset()

print("Loading complete. Starting...")

run_loop(200)
#    
    
MainLoop.export_error()
    
    