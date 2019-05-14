#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:21:22 2019

@author: dzenn
"""

#exec(open("MainRNNLoop.py").read())

try:
    from DynapseRNN import DynapseRNN
except ModuleNotFoundError:
    print("Trying to load Dynapse RNN from outside TransferRNN directory")
    from TransferRNN.DynapseRNN import DynapseRNN
    
#try:
#    from bias_tuning_tools import BiasTools
#except ModuleNotFoundError:
#    from TransferRNN.bias_tuning_tools import BiasTools
    
import numpy as np
import datetime

import pickle



class MainRNNLoop():
    
    def __init__(self, num_inputs = 128, timesteps = 28, backend = "ctxctl", c = None):     
        
        self.RNNController = DynapseRNN(num_inputs=num_inputs, timesteps=timesteps, multiplex_factor=2, backend=backend, c=c, debug=True)
        self.recorded_error = []
        
    def prepare_dataset(self, dataset_path = ""):
        """
            Loads datasets (input, teacher and mnist) into MainRNNLoop internal variables     
            
            Args:
                dataset_path (str, optional) : path to datasets
        """
        
        projection_train = open(dataset_path + 'x_projection_train.pkl', 'rb')
        state_train = open(dataset_path + 'state_train.pkl', 'rb')
        mnist = open(dataset_path + 'mnist.pkl','rb')

        u_1 = pickle._Unpickler(projection_train)
        u_2 = pickle._Unpickler(state_train)
        u_3 = pickle._Unpickler(mnist)
        u_1.encoding = 'latin1'
        u_2.encoding = 'latin1'
        u_3.encoding = 'latin1'
        
        self.projection_train_data = u_1.load()
        self.state_train_data = u_2.load()
#        self.mnist_train, self.mnist_test, self.mnist_validate = u_3.load() # load validation set
        
    def export_error(self, filename = "RNNError.dat"):
        """
            Export mean error log to a file
            
            Args:
                filename (str, optional)
        """
#        f_out = open(str(datetime.datetime.now()) + " " + filename, "wb")        
        f_out = open(filename, "w")
        for err in self.recorded_error:
            f_out.write(str(err) + "\n")
        f_out.close()
        
    def export_conn_log(self, filename = "RNNConnLog.dat"):
        """
            Export recorded connectivity changes
            
            Args:
                filename (str, optional)
        """
#        f_out = open(str(datetime.datetime.now()) + " " + filename, "wb")        
        f_out = open(filename, "w")
        for item in self.RNNController.conn_log:
            f_out.write(str(item[0]) + " " + str(item[1]) + "\n")
        f_out.close()
        
        
    def start(self):
        pass
        
    def run_loop(self, num_images):
        """
            Run the image presentation loop num_images times.
            The loop consists of 6 steps:
                1. Start recording spikes (initialize event filters)
                2. Present the input spike trains (actual realtime image presetation)
                3. Stop recording (clear event filters)
                4. Compute rates based on recorded events by binning spikes by neuron indices and timesteps
                5. Compute the new ternary weight matrix based on the rates and the state_train_data (takes top cam_num largest gradients per neuron with stochastic rounding)
                6. Apply the new weight matrix to the chip
                
            Note:
                Steps 2 and 6 take the most amount of time    
        """
    
        for image_idx in range(num_images):
            print("Start recording")
            self.RNNController.start_recording_spikes()
            print("Showing digit %d" % (image_idx))
            self.RNNController.present_stimulus(self.projection_train_data[image_idx], 2/6)
            print("Stopping the recording")
            self.RNNController.stop_recording_spikes()
            print("Processing recorded evts...")
            rates = self.RNNController.process_recorded_evts()
            print(np.array(rates)/100)
            
            print("Computing gradients...")
            c_grad, mean_error = self.RNNController.update_weight(np.array(rates)/100, (self.state_train_data[image_idx]), learning_rate = 0.01)
            
            self.recorded_error.append(mean_error)
            
            self.RNNController.apply_new_matrix(self.RNNController.w_ternary, False)
            
            print("C_grad: %g, mean_error %g" % (c_grad, mean_error))
            print("Done")        
        

    
    