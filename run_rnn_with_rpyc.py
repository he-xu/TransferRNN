#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 23:45:13 2019

@author: dzenn
"""

#import PyCtxUtils; PyCtxUtils.start_rpyc_server()

import MainRNNLoop

import rpyc

print("Successful import")
self_path = MainRNNLoop.__file__
self_path = self_path[0:self_path.rfind('/')]
execute_bias_load_string = 'exec(open("' + self_path + '/biases/256_biases.py").read())'
execute_rate_tracker_load_string = 'exec(open("' + self_path + '/ctxctl_rate_tracker.py").read())'



c = rpyc.classic.connect("localhost", 1300)
RPYC_TIMEOUT = 1000 #defines a higher timeout
c._config["sync_request_timeout"] = RPYC_TIMEOUT  # Set timeout to higher level
if c:
    print("RPyC connection established")

c.execute(execute_bias_load_string)
c.execute(execute_rate_tracker_load_string)

print("Clearing CAMs...")
c.execute("CtxDynapse.dynapse.clear_cam(1)")
c.execute("CtxDynapse.dynapse.clear_cam(3)")

print("Initializing the main loop")
MainLoop = MainRNNLoop.MainRNNLoop(backend = "rpyc", c = c)
c.namespace['neuron_ids'] = MainLoop.RNNController.neuron_ids
c.namespace['lookup'] = MainLoop.RNNController.rnn_neurons_idx_lookup
c.namespace['timesteps'] = MainLoop.RNNController.timesteps
c.execute("rt = RateTracker(neuron_ids, lookup, timesteps, debug = True)")

print("Loading the dataset")
MainLoop.prepare_dataset("data/")

print("Loading complete. Starting...")

MainLoop.run_loop(100)
#    
    
MainLoop.export_error()
MainLoop.export_conn_log()