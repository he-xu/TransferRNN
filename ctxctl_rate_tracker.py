#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:14:16 2019

@author: dzenn
"""


import CtxDynapse
from time import time

class RateTracker(object):
    def __init__(self, neuron_ids, lookup, timesteps, debug = False):
        
        self.timesteps = timesteps
        self.neuron_ids = neuron_ids
        self.num_neurons = len(neuron_ids)
        self.lookup = lookup
        self.debug = debug

    def start_recording_spikes(self):
        """
            Initializes the event filter
        """
        model = CtxDynapse.model
        self.buf_evt_filter = CtxDynapse.BufferedEventFilter(model, self.neuron_ids)
        evts = self.buf_evt_filter.get_events() # flush the event filter to start a new recording
        self.recording_start_time = time()
        
    def stop_recording_spikes(self):
        """
            Stores all recorded events and clears the event filter
        """
        self.evts = self.buf_evt_filter.get_events()
        self.recording_stop_time = time()
        self.buf_evt_filter.clear()

        
    def process_recorded_evts(self):
        """
            Returns firing rates AND input rates based on recorded events and current
            weight matrix
        """

        lookup = self.lookup
        
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
        
        if self.debug:
            print("Binning...")
        # Placing spikes in bins
        for evt in self.evts:
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
                    
        return rates