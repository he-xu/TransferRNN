#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:09:09 2019

@author: dzenn
"""


from time import clock, sleep, time
from math import exp
import CtxDynapse
import NeuronNeuronConnector
from CtxDynapse import DynapseCamType as SynapseTypes

import sys

sys.path.append('/home/dzenn/anaconda3/envs/ctxctl3.7/lib/python3.7/site-packages')

import numpy as np

class DynapseRNN(object):
    """
       Spike processing class for CortexCotrol for off-chip plasticity by
       monitoring pairs of neurons (e.g. virtual synapses) and tracking some
       parameters for each pair.
       
       The learning rule implemented is reward-gated triplet STDP.       
       STDP weight changes are applied to eligibility traces instead.
       The weights are changed according to the values of the eligibility
       traces at the moment of the reward coming.
       
       The weight change is implemented by adding multiple
       on-chip connections.
    """
        
    def __init__(self, num_inputs, timesteps, multiplex_factor = 1, chip_id_proxy = 3, chip_id = 1, debug = False):
        
        self.creation_time = time()
        self.model = CtxDynapse.model
        self.v_model = CtxDynapse.VirtualModel()
        self.neurons = self.model.get_shadow_state_neurons()
        self.virtual_neurons = self.v_model.get_neurons()
        self.neuron_ids = []
        self.poisson_spike_gen = self.model.get_fpga_modules()[0]
        
        self.buf_evt_filter = None
        self.evts = []
        self.recording_start_time = None
        self.recording_stop_time = None
        
        self.num_inputs = num_inputs
        self.timesteps = timesteps
        
        self.pre_lookup = {}
        self.post_lookup = {}
        self.print = debug
        self.last_pre_timestamp = {}
        self.last_post_timestamp = {}
        self.last_timestamp = {}
        self.last_clock_time = {}
        self.connector = NeuronNeuronConnector.DynapseConnector()
        
        self.learn = True
        
        self.debug = debug

        self.current_weight_matrix = {}
        
        self.record = False
        self.record_da = False
        self.record_timings = False
        self.record_conn_log = False
        
        self.conns_created = 0
        
        self.da_threshold = 1
        
        self.pre_evts = []
        self.post_evts = []
        self.pre_times = []
        self.post_times = []
        self.da_evts = []
        self.pre_diff = []
        self.post_diff = []
        self.conn_log = []
        
        ### GD Learning variables
        
        self.error = None
        self.learning_rate = 0.1
        self.regularizer = 1

        self.multiplex_factor = multiplex_factor
        
        ### Allocating neurons
        
        self.rnn_neurons = [n for n in self.neurons if n.get_chip_id()==chip_id
             and (n.get_neuron_id() + n.get_core_id()*256) < self.num_inputs * self.multiplex_factor]
        
        self.proxy_neurons = [n for n in self.neurons if n.get_chip_id()==chip_id_proxy
             and (n.get_neuron_id() + n.get_core_id()*256) < self.num_inputs*2] 
        
        for n in self.proxy_neurons:
            self.connector.add_connection(self.virtual_neurons[(n.get_neuron_id())],
                                          n,
                                          SynapseTypes.SLOW_EXC)
            self.connector.add_connection(self.virtual_neurons[(n.get_neuron_id())],
                                          n,
                                          SynapseTypes.SLOW_EXC)
            
        for n in self.rnn_neurons:
            self.neuron_ids.append(n.get_neuron_id() + n.get_core_id()*256 + n.get_chip_id()*1024)
            self.connector.add_connection(self.proxy_neurons[(n.get_neuron_id()) % self.num_inputs],
                                          n,
                                          SynapseTypes.SLOW_EXC)
            self.connector.add_connection(self.proxy_neurons[(n.get_neuron_id()) % self.num_inputs],
                                          n,
                                          SynapseTypes.SLOW_EXC)
            self.connector.add_connection(self.proxy_neurons[((n.get_neuron_id()) % self.num_inputs) + self.num_inputs],
                                          n,
                                          SynapseTypes.SLOW_INH)
            self.connector.add_connection(self.proxy_neurons[((n.get_neuron_id()) % self.num_inputs) + self.num_inputs],
                                          n,
                                          SynapseTypes.SLOW_INH)
            
        self.model.apply_diff_state()
        
        self.num_neurons = len(self.neuron_ids)
        
        
        self.P_prev = self.regularizer*np.eye(self.num_neurons)
        
        
        self.poisson_spike_gen.set_chip_id(chip_id_proxy)
            
        if self.debug:
            print("Connected spikegen")
        
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
            
        self.init_weights()
        self.apply_new_matrix(self.w_ternary)
        
        if self.debug:
            print("RNN Init Complete")
            
    def init_weights(self, cam_num=60):
        w_ternary = np.zeros([self.num_neurons, self.num_neurons])
        w_ternary[:,:cam_num//2]=1
        w_ternary[:,cam_num//2:cam_num]=-1
        w_rand = np.random.rand(self.num_neurons, self.num_neurons)
        w_order = np.argsort(w_rand, axis=0)
        self.w_ternary = w_ternary.T[w_order, np.arange(w_order.shape[1])].T
        
        
    def start_recording_spikes(self):
        self.buf_evt_filter = CtxDynapse.BufferedEventFilter(self.model, self.neuron_ids)
        evts = self.buf_evt_filter.get_events()
        self.recording_start_time = time()
        
    def stop_recording_spikes(self):
        self.evts = self.buf_evt_filter.get_events()
        self.recording_stop_time = time()
        self.buf_evt_filter.clear()
        
    def process_recorded_evts(self):
        """
        Returns firing rates AND input rates based on recorded events and current
        weight matrix
        """
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
        
        if len(self.evts) != 0:
            ref_timestamp = self.evts[0].timestamp
        time_bin_size = int((self.recording_stop_time - self.recording_start_time)*1e+06/self.timesteps)
        
        # Placing spikes in bins
        for evt in self.evts:
            n_id = evt.neuron.get_neuron_id() + 256*evt.neuron.get_core_id() + 1024*evt.neuron.get_chip_id()
            idx = self.neuron_ids.index(n_id)
            
            time_bin = (evt.timestamp - ref_timestamp)//time_bin_size
#            print(idx, time_bin)
            if time_bin < self.timesteps:
                rates[idx][time_bin] += 1
        
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
            Applies weight update to the chip
        """
        
        if self.debug:
            print("Applying connectivity changes...")
            
        num_conns_removed = 0
        num_conns_created = 0
        
#        for i in range(len(self.pre_lookup)):
#            for j in range(len(self.post_lookup)):
#                pre_id = self.neuron_ids[i]
#                post_id = self.neuron_ids[j]
#                if (self.current_weight_matrix[(pre_id, post_id)]) != 0:
#                    self.connector.remove_connection(self.neurons[pre_id], self.neurons[post_id])
#                    if i == 0:
#                        print()
#                    num_conns_removed += 1
#                    self.current_weight_matrix[(pre_id, post_id)] -= 1
#                    
#        if self.debug:
#            print("Done.")
#            print("Neuron 0 matrix sum", np.abs(w_ternary[0, :]).sum())
#            print("%d conns removed, %d conns created" % (num_conns_removed, num_conns_created))
#        
#        self.model.apply_diff_state()
#        
#        arr = [i.get_pre_neuron_id() for i in self.neurons[self.neuron_ids[0]].get_cams()]
#        print(arr)
#        
#        for i in range(len(self.pre_lookup)):
#            for j in range(len(self.post_lookup)):
#                pre_id = self.neuron_ids[i]
#                post_id = self.neuron_ids[j]
#                
#                if print_w:
#                    print(w_ternary[j][i], i, j)
#                if (w_ternary[j][i]) != 0:
#                    if w_ternary[j][i] > 0:
#                        self.connector.add_connection(self.neurons[pre_id], self.neurons[post_id], SynapseTypes.SLOW_EXC)
#                        num_conns_created += 1
#                        self.current_weight_matrix[(pre_id, post_id)] += 1
#                    if w_ternary[j][i] < 0:
#                        self.connector.add_connection(self.neurons[pre_id], self.neurons[post_id], SynapseTypes.FAST_INH)
#                        num_conns_created += 1
#                        self.current_weight_matrix[(pre_id, post_id)] += 1
#                    
#        arr = [i.get_pre_neuron_id() for i in self.neurons[self.neuron_ids[0]].get_cams()]
#        print(arr)
        
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
                            self.connector.add_connection(self.neurons[pre_id], self.neurons[post_id], SynapseTypes.FAST_EXC)
                            num_conns_created += 1
                            self.current_weight_matrix[(pre_id, post_id)] += 1
                            
                    elif delta_w < 0:
                        if current_w <= 0:
                            self.connector.add_connection(self.neurons[pre_id], self.neurons[post_id], SynapseTypes.FAST_INH)
                            num_conns_created += 1
                            self.current_weight_matrix[(pre_id, post_id)] -= 1
                            
                    current_w = self.current_weight_matrix[(pre_id, post_id)]
                            
                    
        
        self.model.apply_diff_state()
        
        if self.debug:
            print("Done.")
            print("Neuron 0 matrix sum", np.abs(w_ternary[0, :]).sum())
            print("%d conns removed, %d conns created" % (num_conns_removed, num_conns_created))
                
        
    
    def present_stimulus(self, stim_array, timestep_length):
        """
            Presents the array of rate to the virtual neuron population
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
            P_up = self.P_prev.dot(r_t.dot(r_t.T.dot(self.P_prev)))
            P_down = 1 + r_t.T.dot(self.P_prev.dot(r_t))
            self.P_prev =  self.P_prev - P_up / P_down
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
    return A*exp(-delta_t/tau)
#    print(len(evts))
    
def add_to_dict(dct, key, value):
    if key in dct:
        dct[key].append(value)
    else:
        dct[key] = [value]
        
    
        
        