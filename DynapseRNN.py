#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:09:09 2019

@author: dzenn
"""

#from ctxctl_rl_stdp_connection_multi import RLSTDPConnectionMulti; conn = RLSTDPConnectionMulti([256*4 + 176], [256*4 + 188], debug = True)
#from ctxctl_rl_stdp_connection_multi import RLSTDPConnectionMulti; conn = RLSTDPConnectionMulti([256*4 + 20], [256*4 + 24], debug = True)
#conn.record = True; conn.record_da = True
#conn.record = False; conn.record_da = False
#conn.export_raster(); conn.export_traces()

#conn.record_timings = True
#conn.record_timings = False
#conn.export_diffs()

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
        
    def __init__(self, num_inputs, timesteps, multiplex_factor = 1, chip_id = 1, debug = False):
        
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
        self.P_prev = None  ##TODO: init P_prev
        self.multiplex_factor = multiplex_factor
        
        ### Allocating neurons
        
        self.rnn_neurons = [n for n in self.neurons if n.get_chip_id()==chip_id
             and n.get_neuron_id() + n.get_core_id()*256 < self.num_inputs * self.multiplex_factor] 
        
        for n in self.rnn_neurons:
            self.neuron_ids.append(n.get_neuron_id() + n.get_core_id()*256 + n.get_chip_id()*1024)
            self.connector.add_connection(self.virtual_neurons[(n.get_neuron_id() + n.get_core_id()*256) % self.num_inputs],
                                          n,
                                          SynapseTypes.SLOW_EXC)
            
        self.model.apply_diff_state()
        
        
        self.poisson_spike_gen.set_chip_id(chip_id)
            
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
            
        if self.debug:
            print("RNN Init Complete")
            
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
        for i in range(len(self.neuron_ids)):
            rates.append([])
            input_rates.append([])
            for ts in range(self.timesteps):
                rates[i].append(0)
                input_rates[i].append(0)
        
        ref_timestamp = self.evts[0].timestamp
        time_bin_size = int((self.recording_stop_time - self.recording_start_time)*1e+06/self.timesteps)
        
        # Placing spikes in bins
        for evt in self.evts:
            n_id = evt.neuron.get_neuron_id() + 256*evt.neuron.get_core_id() + 1024*evt.neuron.get_chip_id()
            idx = self.neuron_ids.index(n_id)
            
            time_bin = (evt.timestamp - ref_timestamp)//time_bin_size
            rates[idx][time_bin] += 1
        
        # Normalizing spike counts to rates
        for i in range(len(self.neuron_ids)):
            for ts in range(self.timesteps):
                rates[i][ts] = rates[i][ts]/(time_bin_size*1e+06)
        
        # Computing weighted input rate sums
        for i in range(len(self.neuron_ids)):
            pre_id = self.neuron_ids[i]
            for post_id in self.post_lookup[pre_id]:
                for ts in range(self.timesteps):
                    input_rates[self.neuron_ids.index(post_id)][ts] += rates[i][ts]*self.current_weight_matrix[(pre_id, post_id)]
                    
        if self.debug:
            print("Returning rates...")
                    
        return rates, input_rates
    
    
    def apply_new_matrix(self, matrix):
        """
            Applies weight update to the chip
        """
        
        if self.debug:
            print("Applying connectivity changes...")
        
        for i in len(self.pre_lookup):
            for j in len(self.post_lookup):
                pre_id = self.neuron_ids[i]
                post_id = self.neuron_ids[j]
                current_w = self.current_weight_matrix[(pre_id, post_id)]
                delta_w = matrix[i][j] - current_w
                
                
                for conn_idx in range(abs(delta_w)):
                    if delta_w > 0:
                        if current_w < 0:
                            self.connector.remove_connection(pre_id, post_id)
                        else:
                            self.connector.add_connection(pre_id, post_id, SynapseTypes.FAST_EXC)
                            
                    elif delta_w < 0:
                        if current_w <= 0:
                            self.connector.add_connection(pre_id, post_id, SynapseTypes.FAST_INH)
                        else:
                            self.connector.remove_connection(pre_id, post_id)
                            
                    self.current_weight_matrix[(pre_id, post_id)] = matrix[i][j]
        
        self.model.apply_diff_state()
        
        if self.debug:
            print("Done.")
                
        
    
    def present_stimulus(self, stim_array, timestep_length):
        """
            Presents the array of rate to the virtual neuron population
        """
        
        if self.debug:
            print("Presenting the digit...")
        
        self.poisson_spike_gen.start()
        
        for ts in range(self.timesteps):        
            for i in range(len(self.neuron_ids)):
                self.poisson_spike_gen.write_poisson_rate_hz(i, abs(stim_array[ts, i]*10))
            
            sleep(timestep_length)
            
        for i in range(len(self.neuron_ids)):
            self.poisson_spike_gen.write_poisson_rate_hz(i, 0)
            
        self.poisson_spike_gen.stop()
        
        if self.debug:
            print("Done.")
        
    def ternarize(w_new, cam_num):
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

    def update_weight(self, rate_psc, rate_teacher, w_ternary, cam_num=63, learning_rate=0.1):
        """
            Returns:
                w_ternary : new on-chip connectivity matrix
                c_grad : increase\decrease global activity level
        """
        rate_recurrent = w_ternary.dot(rate_psc)
        rate_teacher_tile = np.tile(rate_teacher, (2,1))
        self.error = rate_recurrent - rate_teacher_tile
        d_w = 0
        for t in range(self.timesteps):
            r_t = rate_psc[:, t][:,np.newaxis]
            P_up = P_prev.dot(r_t.dot(r_t.T.dot(P_prev)))
            P_down = 1 + r_t.T.dot(P_prev.dot(r_t))
            P_t =  P_prev - P_up / P_down
            e_t = error[:, t][:,np.newaxis]
            d_w += e_t.dot(r_t.T.dot(P_t))
        d_w = d_w / num_timesteps
        w_new = w_ternary - learning_rate*d_w
        w_ternary = ternarize(w_new, cam_num)
        norm_ratio = np.linalg.norm(w_new, 'fro')/np.linalg.norm(w_ternary, 'fro')
        if norm_ratio > 1:
            c_grad = 1
        else:
            c_grad = -1
        return w_ternary, c_grad
        
    
    def connect_spikegen(self, chip_id, syn_type_int):
        """ Creates connections from virtual neuron 1 to all neurons of the selected chip
        using selected syn_type.
        
        Args:
            chip_id (int): stimulated chip ID
            syn_type_int (int): integer synapse type to be converted to DynapseCamType withing the method
        """
        
        self.spikegen_target_chip = chip_id
        
        if syn_type_int == 0:
            syn_type = SynTypes.SLOW_INH
        elif syn_type_int == 1:
            syn_type = SynTypes.FAST_INH
        elif syn_type_int == 2:
            syn_type = SynTypes.SLOW_EXC
        elif syn_type_int == 3:
            syn_type = SynTypes.FAST_EXC
        else:
            print("Unable syn type, please try again")
            return
        
        for n in range(1024):
            self.connector.add_connection(pre=self.virtual_neurons[1],
                                          post=self.neurons[n + 1024*chip_id],
                                          synapse_type=syn_type)
        
        self.model.apply_diff_state()
        
        
        
                

def relaxate(A, tau, delta_t):
    return A*exp(-delta_t/tau)
#    print(len(evts))
    
def add_to_dict(dct, key, value):
    if key in dct:
        dct[key].append(value)
    else:
        dct[key] = [value]
        
    
        
        