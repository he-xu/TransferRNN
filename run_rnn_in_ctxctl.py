#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:23:18 2019

@author: dzenn
"""

#import TransferRNN.run_rnn_in_ctxctl

import sys
sys.path.append('/home/dzenn/anaconda3/envs/ctxctl/lib/python3.7/site-packages')

from TransferRNN.MainRNNLoop import MainRNNLoop
from TransferRNN.bias_tuning_tools import BiasTools



print("Successful import")
    
bt = BiasTools()
exec(open("TransferRNN/biases/256_biases.py").read())
#bt.copy_biases(4,5)
#bt.copy_biases(4,12)
print("Clearing CAMs...")
bt.clear_cams(1)
bt.clear_cams(3)

print("Initializing the main loop")
MainLoop = MainRNNLoop(backend = "ctxctl")

print("Loading the dataset")
MainLoop.prepare_dataset("TransferRNN/data/")

print("Loading complete. Starting...")

MainLoop.run_loop(100)
#    
    
MainLoop.export_error()
MainLoop.export_conn_log()