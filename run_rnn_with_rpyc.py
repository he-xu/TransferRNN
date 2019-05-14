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


c = rpyc.classic.connect("localhost", 1300)
if c:
    print("RPyC connection established")

c.execute(execute_bias_load_string)

#bt.copy_biases(4,5)
#bt.copy_biases(4,12)
print("Clearing CAMs...")
c.modules.CtxDynapse.dynapse.clear_cam(1)
c.modules.CtxDynapse.dynapse.clear_cam(3)

print("Initializing the main loop")
MainLoop = MainRNNLoop.MainRNNLoop(backend = "rpyc", c = c)

print("Loading the dataset")
MainLoop.prepare_dataset("data/")

print("Loading complete. Starting...")

MainLoop.run_loop(100)
#    
    
MainLoop.export_error()
MainLoop.export_conn_log()