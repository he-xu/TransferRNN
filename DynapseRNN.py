#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:09:09 2019

@author: dzenn
"""


from time import clock, sleep, time
from math import exp

try:
    import CtxDynapse
    import NeuronNeuronConnector
    from CtxDynapse import DynapseCamType as SynapseType
except ModuleNotFoundError:
    import rpyc
    print("CtxDynapse module not imported. Expecting to run in RPyC mode")

import numpy as np

class DynapseRNN(object):
    """
        A controller class for learning-in-the-loop RNN on the Dynapse chip using
        CortexControl Python API.
        
        Weight update is defined by FORCE algorithm and performed by
        generating a ternary weight matrix after gradient descent with
        stochastic rounding within the limitation of number of inputs used per neuron.
        
        The controller places the RNN on one chip and a proxy input population on another.
        
        The controller can be used both through rpyc or directly in CTXCTL console.
        
    """
        
    def __init__(self, num_inputs, timesteps, multiplex_factor = 1, chip_id_proxy = 3, chip_id = 1, backend = "ctxctl", c = None, debug = False):
        
        """
            Args:
                num_inputs (int) : number of input channels (i.e. spike generators)
                timesteps (int) : length of each input sequence
                multiplex_factor (int, optional) : number of silicon neurons on the chip
                                         which represent a single logical neuron
                                         (i.e. multiplies the size of the RNN)
                chip_id_proxy (int, optional) : index of the chip where the proxy input
                                         population is located (should not be the same
                                         as the RNN chip)
                chip_id (int, optional) : index of the chip where the RNN is located
                backend (str, optional) : sets whether the controller will use RPyC
                                         or try to import modules directly in the console
                c (RPyC connection, optional) : RPyC connection object, expected if "rpyc" is
                                         set as backend
                debug (bool, optional) : print additional debug info
        """
        
        self.creation_time = time()
        
        if backend == "ctxctl":
            self.model = CtxDynapse.model
            self.v_model = CtxDynapse.VirtualModel()
            self.connector = NeuronNeuronConnector.DynapseConnector()
            self.SynapseType = SynapseType
        elif backend == "rpyc":
            if c is not None:
                self.c = c
                self.model = c.modules.CtxDynapse.model
                self.v_model = c.modules.CtxDynapse.VirtualModel()
                self.connector = c.modules.NeuronNeuronConnector.DynapseConnector()
                self.SynapseType = c.modules.CtxDynapse.DynapseCamType
#                c.execute("from time import time, clock, sleep")
#                c.execute("import CtxDynapse")
            else:
                raise ValueError("Selected backend is 'rpyc' but no connection object is given to DynapseRNN")
        else:
            raise ValueError("Unknown backend identifier '" + backend + "'. Use 'ctxctl' or 'rpyc' only.")
        
        self.backend = backend
        
        self.neurons = self.model.get_shadow_state_neurons()
        self.virtual_neurons = self.v_model.get_neurons()
        self.neuron_ids = []
        self.poisson_spike_gen = self.model.get_fpga_modules()[0]
        
        self.buf_evt_filter = None
        self.evts = []
        self.recording_start_time = None
        self.recording_stop_time = None
        
        self.teleported_functions = [] # list of functions teleported with rpyc
        
        self.num_inputs = num_inputs
        self.timesteps = timesteps
        
        self.pre_lookup = {}
        self.post_lookup = {}
        self.print = debug
        self.last_pre_timestamp = {}
        self.last_post_timestamp = {}
        self.last_timestamp = {}
        self.last_clock_time = {}
        
        
        self.learn = True
        
        self.debug = debug

        self.current_weight_matrix = {}
        
        self.conns_created = 0
              
        self.pre_evts = []
        self.post_evts = []
        self.pre_times = []
        self.post_times = []
        self.da_evts = []
        self.pre_diff = []
        self.post_diff = []
        self.conn_log = []
        
        ### GD Learning parameters
        
        self.error = None
        self.learning_rate = 0.1
        self.regularizer = 1

        self.multiplex_factor = multiplex_factor
        
        ### Allocating neurons
        if self.debug:
            print("Allocating populations")
        
        self.rnn_neurons = [n for n in self.neurons if n.get_chip_id()==chip_id
             and (n.get_neuron_id() + n.get_core_id()*256) < self.num_inputs * self.multiplex_factor]
        
        self.proxy_neurons = [n for n in self.neurons if n.get_chip_id()==chip_id_proxy
             and (n.get_neuron_id() + n.get_core_id()*256) < self.num_inputs*2] 
        
        self.rnn_neurons_idx_lookup = {}
        i = 0
        for neuron in self.rnn_neurons:
            self.rnn_neurons_idx_lookup.update({neuron.get_neuron_id() + 256*neuron.get_core_id() + 1024*neuron.get_chip_id() : i})
            i += 1
        ### Creating spikegen -> proxy_neurons connections
        
        for n in self.proxy_neurons:
            self.connector.add_connection(self.virtual_neurons[(n.get_neuron_id())],
                                          n,
                                          self.SynapseType.SLOW_EXC)
            self.connector.add_connection(self.virtual_neurons[(n.get_neuron_id())],
                                          n,
                                          self.SynapseType.SLOW_EXC)
            
        if self.debug:
            print("Connected spikegen")
        
        ### Creating proxy_neurons -> rnn_neurons connections
        
        for n in self.rnn_neurons:
            self.neuron_ids.append(n.get_neuron_id() + n.get_core_id()*256 + n.get_chip_id()*1024)
            self.connector.add_connection(self.proxy_neurons[(n.get_neuron_id()) % self.num_inputs],
                                          n,
                                          self.SynapseType.SLOW_EXC)
            self.connector.add_connection(self.proxy_neurons[(n.get_neuron_id()) % self.num_inputs],
                                          n,
                                          self.SynapseType.SLOW_EXC)
            self.connector.add_connection(self.proxy_neurons[((n.get_neuron_id()) % self.num_inputs) + self.num_inputs],
                                          n,
                                          self.SynapseType.FAST_INH)
            self.connector.add_connection(self.proxy_neurons[((n.get_neuron_id()) % self.num_inputs) + self.num_inputs],
                                          n,
                                          self.SynapseType.FAST_INH)
            
        self.model.apply_diff_state()        
        self.num_neurons = len(self.neuron_ids)    
        
#        self.P_prev = self.regularizer*np.eye(self.num_neurons)         
        self.poisson_spike_gen.set_chip_id(chip_id_proxy)
            
        if self.debug:
            print("Connected proxy population")
        
        pre_id_list = []
        post_id_list = []
        
        for i in range(self.num_inputs * self.multiplex_factor):
            for j in range(self.num_inputs * self.multiplex_factor):
                pre_id_list.append(self.neuron_ids[i])
                post_id_list.append(self.neuron_ids[j])
                
        
        ## Prepare connectivity matrix
        for pre_id, post_id in zip(pre_id_list, post_id_list):
            add_to_dict(self.post_lookup, pre_id, post_id)
            add_to_dict(self.pre_lookup, post_id, pre_id)
            self.last_pre_timestamp[(pre_id, post_id)] = 0
            self.last_post_timestamp[(pre_id, post_id)] = 0
            self.last_timestamp[(pre_id, post_id)] = 0
            self.last_clock_time[(pre_id, post_id)] = 0

            self.current_weight_matrix[(pre_id, post_id)] = 0
            
#        self.init_weights()
        self.w_ternary = np.zeros([self.num_neurons, self.num_neurons])
        self.apply_new_matrix(self.w_ternary)
        
#        if self.backend == 'rpyc':
#            self.teleported_functions.append(rpyc.utils.classic.teleport_function(c, self.start_recording_spikes))
#            self.teleported_functions.append(rpyc.utils.classic.teleport_function(c, self.stop_recording_spikes))
#            self.teleported_functions.append(rpyc.utils.classic.teleport_function(c, self.process_recorded_evts))
#            self.start_recording_spikes = lambda: self.teleported_functions[0](self)
#            self.stop_recording_spikes = lambda: self.teleported_functions[1](self)
#            self.process_recorded_evts = lambda: self.teleported_functions[2](self)
        

        
        if self.debug:
            print("RNN Init Complete")
            
    def init_weights(self, cam_num=60):
        """
            Random ternary weight initialization
        """
        w_ternary = np.zeros([self.num_neurons, self.num_neurons])
        w_ternary[:,:cam_num//2]=1
        w_ternary[:,cam_num//2:cam_num]=-1
        w_rand = np.random.rand(self.num_neurons, self.num_neurons)
        w_order = np.argsort(w_rand, axis=0)
        self.w_ternary = w_ternary.T[w_order, np.arange(w_order.shape[1])].T
        
        
    def start_recording_spikes(self):
        """
            Initializes the event filter
        """
        if self.backend == 'rpyc':
            self.c.execute('rt.start_recording_spikes()')
        else:                   
            model = CtxDynapse.model
            self.buf_evt_filter = CtxDynapse.BufferedEventFilter(model, self.neuron_ids)
            evts = self.buf_evt_filter.get_events() # flush the event filter to start a new recording
            self.recording_start_time = time()
        
    def stop_recording_spikes(self):
        """
            Stores all recorded events and clears the event filter
        """
        if self.backend == 'rpyc':
            self.c.execute('rt.stop_recording_spikes()')
        else:
            self.evts = self.buf_evt_filter.get_events()
            self.recording_stop_time = time()
            self.buf_evt_filter.clear()

        
    def process_recorded_evts(self):
        """
            Returns firing rates AND input rates based on recorded events and current
            weight matrix
        """
        if self.backend == 'rpyc':
            self.c.execute('rates = rt.process_recorded_evts()')
            return self.c.namespace['rates']
        else:
            
            lookup = self.rnn_neurons_idx_lookup
            evts = self.evts
            rates = []
            input_rates = []
            
            if self.debug:
                print("Counting the spikes...")
            
            # Preparing arrays and helper variables
            for i in range(self.num_neurons):
                rates.append([])
                input_rates.append([])
                for ts in range(self.timesteps):
                    rates[i].append(0)
                    input_rates[i].append(0)
            
            if len(evts) != 0:
                ref_timestamp = evts[0].timestamp
            time_bin_size = int((self.recording_stop_time - self.recording_start_time)*1e+06/self.timesteps)
            
            if self.debug:
                print("Binning...")
            # Placing spikes in bins
            for evt in evts:
                n_id = evt.neuron.get_neuron_id() + 256*evt.neuron.get_core_id() + 1024*evt.neuron.get_chip_id()
                idx = lookup[n_id]
                
                time_bin = (evt.timestamp - ref_timestamp)//time_bin_size
    #            print(idx, time_bin)
                if time_bin < self.timesteps:
                    rates[idx][time_bin] += 1
            
            if self.debug:
                print("Normalizing...")
            # Normalizing spike counts to rates
            for i in range(self.num_neurons):
                for ts in range(self.timesteps):
                    rates[i][ts] = rates[i][ts]/(time_bin_size/1e+06)
            
    #        # Computing weighted input rate sums
    #        for i in range(self.num_neurons):
    #            pre_id = self.neuron_ids[i]
    #            for post_id in self.post_lookup[pre_id]:
    #                for ts in range(self.timesteps):
    #                    input_rates[self.neuron_ids.index(post_id)][ts] += rates[i][ts]*self.current_weight_matrix[(pre_id, post_id)]
                        
            if self.debug:
                print("Returning rates...")
                        
            return rates #, input_rates
    
    
    def apply_new_matrix(self, w_ternary, print_w = False):
        """
            Applies the new weight matrix to the chip
        """
        
        if self.debug:
            print("Applying connectivity changes...")
            
        num_conns_removed = 0
        num_conns_created = 0
    
        
        for i in range(len(self.pre_lookup)):
            for j in range(len(self.post_lookup)):
                pre_id = self.neuron_ids[i]
                post_id = self.neuron_ids[j]
                current_w = self.current_weight_matrix[(pre_id, post_id)]
                delta_w = w_ternary[j][i] - current_w
                
#                if self.debug:
#                    print(w_ternary[j][i], current_w)
#                    print("Delta: ", int(abs(delta_w)))
                
                for conn_idx in range(int(abs(delta_w))):
                    if print_w:
                        print("removal phase")
                        print(delta_w, current_w, w_ternary[j][i], i, j)
                    
                    if delta_w > 0:
                        if current_w < 0:
                            self.connector.remove_connection(self.neurons[pre_id], self.neurons[post_id])
                            num_conns_removed += 1
                            self.current_weight_matrix[(pre_id, post_id)] += 1
                            
                    elif delta_w < 0:
                        if current_w > 0:
                            self.connector.remove_connection(self.neurons[pre_id], self.neurons[post_id])
                            num_conns_removed += 1
                            self.current_weight_matrix[(pre_id, post_id)] -= 1
                    
                    current_w = self.current_weight_matrix[(pre_id, post_id)]
                            

                    
        for i in range(len(self.pre_lookup)):
            for j in range(len(self.post_lookup)):
                pre_id = self.neuron_ids[i]
                post_id = self.neuron_ids[j]
                current_w = self.current_weight_matrix[(pre_id, post_id)]
                delta_w = w_ternary[j][i] - current_w
                
#                if self.debug:
#                    print(w_ternary[j][i], current_w)
#                    print("Delta: ", int(abs(delta_w)))
                
                for conn_idx in range(int(abs(delta_w))):
                    if print_w:
                        print("addition phase")
                        print(delta_w, current_w, w_ternary[j][i], i, j)
                    
                    if delta_w > 0:
                        if current_w >= 0:
                            self.connector.add_connection(self.neurons[pre_id], self.neurons[post_id], self.SynapseType.FAST_EXC)
                            num_conns_created += 1
                            self.current_weight_matrix[(pre_id, post_id)] += 1
                            
                    elif delta_w < 0:
                        if current_w <= 0:
                            self.connector.add_connection(self.neurons[pre_id], self.neurons[post_id], self.SynapseType.FAST_INH)
                            num_conns_created += 1
                            self.current_weight_matrix[(pre_id, post_id)] -= 1
                            
                    current_w = self.current_weight_matrix[(pre_id, post_id)]
                            
                    
        
        self.model.apply_diff_state()
        
        if self.debug:
            print("Done.")
            print("Neuron 0 matrix sum", np.abs(w_ternary[0, :]).sum())
            print("%d conns removed, %d conns created" % (num_conns_removed, num_conns_created))
            
        self.conn_log.append([num_conns_removed, num_conns_created])
                
        
    
    def present_stimulus(self, stim_array, timestep_length):
        """
            Presents the array of rates to the virtual neuron population
        """
        
        if self.debug:
            print("Presenting the digit...")
        
        self.poisson_spike_gen.start()
        
        for ts in range(self.timesteps):        
            for i in range(self.num_inputs):
                rate = stim_array[ts, i]*100
                if rate >= 0:
                    self.poisson_spike_gen.write_poisson_rate_hz(i, rate)
                else:
                    self.poisson_spike_gen.write_poisson_rate_hz(i + self.num_inputs, abs(rate))
            
            sleep(timestep_length)
            
        for i in range(self.num_inputs):
            self.poisson_spike_gen.write_poisson_rate_hz(i, 0)
            
        self.poisson_spike_gen.stop()
        
        if self.debug:
            print("Done.")
        
    def ternarize(self, w_new, cam_num):
        """
            
        """
        w_order = np.argsort(np.abs(w_new.T), axis=0)
        w_sorted = w_new.T[w_order, np.arange(w_order.shape[1])]
        w_sorted[:-cam_num, :]=0
        w_order_order = np.argsort(w_order, axis=0)
        w_undone = w_sorted[w_order_order, np.arange(w_order_order.shape[1])].T
        w_undone[w_undone>0] = 1
        w_undone[w_undone<0] = -1
        return w_undone
    
    
    def stochastic_round(self, w_ternary, d_w, cam_num):
        """
            Stochastically rounds the ternary connectivity matrix 
        """
        w_new = w_ternary - d_w
        w_uniform = np.random.uniform(size=d_w.shape)
        d_w_rounded = ((w_uniform < np.abs(d_w))*np.sign(d_w)).astype(np.int)
        w_new_rounded = w_ternary - d_w_rounded
        w_new_rounded[w_new_rounded>1] = 1
        w_new_rounded[w_new_rounded<-1] = -1
        
        w_order = np.argsort(np.abs(w_new.T), axis=0)
        w_new_rounded_sorted = w_new_rounded.T[w_order, np.arange(w_order.shape[1])]
        num_neuron = w_order.shape[1]
        for idx_post in range(num_neuron):
            cam_used = 0
            for idx_pre in range(num_neuron):
                w_ij = w_new_rounded_sorted[-idx_pre, idx_post]
                if np.abs(w_ij) > 0.1:
                    cam_used += 1
                if cam_used >= cam_num:
                    w_new_rounded_sorted[:-idx_pre, idx_post] = 0
                    break
        w_order_order = np.argsort(w_order, axis=0)
        w_undone = w_new_rounded_sorted[w_order_order, np.arange(w_order_order.shape[1])].T
        return w_undone

    def update_weight(self, rate_psc, rate_teacher, cam_num=60, learning_rate=0.1):
        """
            Generates the new ternary connectivity matrix based on measured on-chip rates and teacher signal
            
            Args:
                rate_psc (numpy array) : array of rates of shape (num_neurons, timesteps)
                rate_teacher (numpy array) : array of teacher rates of shape (num_inputs, timesteps)
                cam_num (int, optional) : maximum number of CAMs used by each neurons
                learning_rate (float, optional) : scaler of the weight change gradients
                
            Returns:
                w_ternary : new on-chip connectivity matrix
                c_grad : increase\decrease global activity level
        """
        rate_recurrent = self.w_ternary.dot(rate_psc)
        rate_teacher_tile = np.tile(rate_teacher.T, (self.multiplex_factor,1))
        self.error = rate_recurrent - rate_teacher_tile
        d_w = 0
        for t in range(self.timesteps):
            r_t = rate_psc[:, t][:,np.newaxis]
#            P_up = self.P_prev.dot(r_t.dot(r_t.T.dot(self.P_prev)))
#            P_down = 1 + r_t.T.dot(self.P_prev.dot(r_t))
#            self.P_prev =  self.P_prev - P_up / P_down
            e_t = self.error[:, t][:,np.newaxis]
#            d_w += e_t.dot(r_t.T.dot(self.P_prev))
            d_w += e_t.dot(r_t.T)
        d_w = d_w / self.timesteps
        w_new = self.w_ternary - learning_rate*d_w
        norm_ratio = np.linalg.norm(w_new, 'fro')/np.linalg.norm(self.w_ternary, 'fro')
        self.w_ternary = self.stochastic_round(self.w_ternary, learning_rate*d_w, cam_num)
        
        #self.w_ternary = self.ternarize(w_new, cam_num)
        
        print(d_w.mean(), d_w.max(), d_w.min())
        print(rate_recurrent.mean(), rate_teacher.mean())

        if norm_ratio > 1:
            c_grad = 1
        else:
            c_grad = -1
        return c_grad, np.abs(self.error).mean()        
        
        
                

def relaxate(A, tau, delta_t):
    """
        Computes the exponential
    """
    return A*exp(-delta_t/tau)
#    print(len(evts))
    
def add_to_dict(dct, key, value):
    """
        A tool to add elements to dictionaries of lists.
        
        Appends a value to the list dct[key], otherwise creates it.
    """
    if key in dct:
        dct[key].append(value)
    else:
        dct[key] = [value]
        
    
        
        