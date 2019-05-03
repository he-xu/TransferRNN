#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:51:51 2018

Set of tools to simplify the bias tuning process of the DYNAP-SE chip
using aiCTX's cortexcontrol.

Load and instantiate Bias Tuning Tools (BiTTs) in cortexcontrol console:
    >>> import bias_tuning_tools
    >>> bt = bias_tuning_tools.BiasTools()
    
Several actions available:
    
    Monitoring I_mem:
        Use bt.monitor(core_id, neuron_id) to output I_mem to the respective DYNAP-SE output
    
    Stimulating a core with steps of DC current:
        Use bt.dc_steps(core_id, period, time_interval, coarse_val) to run steps of current with
        specified period and aplitude for the duration of the specified time interval.
        
    Stimulating a chip with a periodic spike train:
        Step 1:
            Use bt.connect_spikegen(chip_id, syn_type) to select the chip and the synapse to use.
            For convinience, syn_type is integer, 0 is SLOW_INH, 1 is FAST_INH, 2 is SLOW_EXC, 3 is FAST_EXC
        Step 2:
            Use bt.set_rate(rate) to set the rate of the spike train and start spikegen.
            
            You can always start and stop spiken with bt.spikegen.start() and bt.spikegen.stop()
            and check the status with bt.spikegen.is_running()
            
    Saving and loading biases with wrapper functions:
        bt.save_biases(filename) and bt.load_biases(filename).
        IMPORTANT: Make sure the filename ends with *.py
        ##TODO: Add a check for "*.py"
    
    Copying biases:
        Use bt.copy_biases(source_core, target_core) to copy the full set of biases from one
        core to another.

@author: dzenn
"""


# import bias_tuning_tools; bt = bias_tuning_tools.BiasTools();
# import imp; imp.reload(bias_tuning_tools);

#import sys
#sys.path.append('/home/dzenn/anaconda3/envs/ctxctl3.7/lib/python3.7/site-packages')
#sys.path.append('/home/dzenn/anaconda3/envs/ctxctl/lib/python3.7/site-packages')
#sys.path.append('/home/theiera/gitlab/NCS/CTXCTL/cortexcontrol')

#import numpy as np
from time import sleep, clock
import CtxDynapse
from NeuronNeuronConnector import DynapseConnector
import PyCtxUtils
from CtxDynapse import DynapseCamType as SynTypes
#import numpy as np


class BiasTools(object):

    def __init__(self):
        
        self.model = CtxDynapse.model
        self.virtual_model = CtxDynapse.VirtualModel()
        self.groups = self.model.get_bias_groups()
        self.neurons = self.model.get_shadow_state_neurons()
        self.virtual_neurons = self.virtual_model.get_neurons()
        self.buf_evt_filter = None
        self.spikes = []
        self.event_filters = []
        self.core_rate = None
        self.coarse_value = None
        self.coarse_value_set = False
        self.connector = DynapseConnector()
        self.poisson_spike_gen = self.model.get_fpga_modules()[0]
        self.spikegen = self.model.get_fpga_modules()[1]
        self.syn_types = SynTypes
        self.spikegen_target_chip = None
        
        
    def off_core(self, core_id):
        """
        Switch the core off using by setting TAU1 to a large value
        
        Args:
            core_id (int): core index [0-15]
        """

        self.groups[core_id].set_bias("IF_TAU1_N", 100, 7)
        
    def save_biases(self, filename):
        """
        A wrapper to save the biases to a file
        
        Args:
            filename (str): filename should end *.py
        """
        PyCtxUtils.save_biases(filename)
        
    def load_biases(self, filename):
        """
        A wrapper to load the biases from file
        
        Args:
            filename (str): filename should end *.py
        """        
        exec(open(filename).read())

    def clear_all_cams(self):
        """
        Clear cams of the whole board
        """
        for chip_id in range(4):
            CtxDynapse.dynapse.clear_cam(chip_id)
            
    def clear_cams(self, chip_id):
        """
        Clear cams of the specified chip
        
        Args:
            chip_id (int): chip index
        """
        CtxDynapse.dynapse.clear_cam(chip_id)

    def clear_all_srams(self):
        """
        Clear srams of the whole board
        """
        for chip_id in range(4):
            CtxDynapse.dynapse.clear_sram(chip_id)
            
    def clear_sram(self, chip_id):
        """
        Clear srams of the specified chip
        
        Args:
            chip_id (int): chip index
        """
        CtxDynapse.dynapse.clear_sram(chip_id)

    
    def copy_biases(self, source_core, target_core):
        """
        Copies the full set of biases from one core to another
        
        Args:
            source_core (int): core_id from 0 to 15 from where to copy the biases
            target_core (int): core_id from 0 to 15 where to write the biases
        """
        
        source_biases = self.groups[source_core].get_biases()
  
        for bias in source_biases:
            self.groups[target_core].set_bias(bias.get_bias_name(), bias.get_fine_value(), bias.get_coarse_value())

        
        
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
            
    def disconnect_spikegen(self):
        """
        Removes all connections from the spikegen to physical neurons.
        
        ##TODO: Seems not to work, needs fixing
        """
        
        self.spikegen_target_chip = None
        
        for n in range(1024):
            self.connector.remove_sending_connections(self.virtual_neurons[n])
        
        self.model.apply_diff_state()
            

    def set_rate(self, rate):
        """
        Sets spiking rate of the spikegen and starts it.
        
        Args:
            rate (int): Spiking rate in Hz, can't be lower ~5 Hz
        """
        
        self.spikegen.stop()
        

        
        isi_base = 900
        unit_mult = isi_base/90
        
        sleep((rate**(-1)))
        
        fpga_event = CtxDynapse.FpgaSpikeEvent()
        fpga_event.core_mask = 15
        fpga_event.target_chip = self.spikegen_target_chip
        fpga_event.neuron_id = 1
        fpga_event.isi = int(((rate*1e-6)**(-1))/unit_mult)
        
        
        self.spikegen.set_variable_isi(False)
        self.spikegen.preload_stimulus([fpga_event])
        self.spikegen.set_isi(int(((rate*1e-6)**(-1))/unit_mult))
        self.spikegen.set_isi_multiplier(isi_base)
        self.spikegen.set_repeat_mode(True)

        self.spikegen.start()


       
    def dc_steps(self, core_id, period, time_interval, coarse_val):
        """ Create square steps of DC current using the IF_DC_P bias with
        the specified period and within the specified time_interval, with coarse_val amplitude.
        
        Args:
            core_id (int): target core
            period (int): period of DC steps, in seconds
            time_interval (int): time interval for the method to run, in seconds
            coarse_val (int): amplitude of DC steps set by the coarse value of the IF_DC_P bias (fine value is set to 128)
        """
        start_time = clock()
        while (clock() - start_time < time_interval):
            self.groups[core_id].set_bias("IF_DC_P", 128, coarse_val)
            sleep(period)
            self.groups[core_id].set_bias("IF_DC_P", 0, 0)
            sleep(period)
            
    def get_core_rate(self, core_id, time_interval):
        """ Create square steps of DC current using the IF_DC_P bias with
        the specified period and within the specified time_interval, with coarse_val amplitude.
        
        Args:
            core_id (int): target core
            period (int): period of DC steps, in seconds
            time_interval (int): time interval for the method to run, in seconds
            coarse_val (int): amplitude of DC steps set by the coarse value of the IF_DC_P bias (fine value is set to 128)
        """
        buf_evt_filter = CtxDynapse.BufferedEventFilter(self.model, [idx for idx in range(core_id*256, core_id*256 + 256)])
        start_time = clock()
        
        while (clock() - start_time < time_interval):
            evts = buf_evt_filter.get_events()
            print("Core %d average rate is %g Hz" % (core_id, len(evts)/256))
            sleep(1)
            
        buf_evt_filter.clear()
        
    def get_rates(self, n_ids = None, measurement_period = 1):
        
        if n_ids is None:
            n_ids = [l for l in range(len(self.neurons))]
        
        buf_evt_filter = CtxDynapse.BufferedEventFilter(self.model, n_ids)
        evts = buf_evt_filter.get_events()
        sleep(measurement_period)
        evts = buf_evt_filter.get_events()
        buf_evt_filter.clear()
        
        rates = [0 for l in range(len(self.neurons))]
        if len(evts) != 0:
            for spike in evts:
                rates[spike.neuron.get_neuron_id() + 256*spike.neuron.get_core_id() + 1024*spike.neuron.get_core_id()] += 1
            
        for l in range(len(self.neurons)):
            rates[l] = rates[l]/measurement_period
            
        return rates
        
        
    def r(self):
        print(self.get_rates([idx for idx in range(0, 10)]))
        
        
        
        
    def get_core_rate_ts(self, core_id, time_interval):

        buf_evt_filter = CtxDynapse.BufferedEventFilter(self.model, [idx for idx in range(core_id*256, core_id*256 + 25)])
        start_time = clock()
        
        while (clock() - start_time < time_interval):
            evts = buf_evt_filter.get_events()
            if len(evts) != 0:                
                print("Core %d average rate is %g Hz" % (core_id, len(evts)/(((evts[len(evts)-1].timestamp - evts[0].timestamp) / 1e6)*25)))
            sleep(1)
            
        buf_evt_filter.clear()
            
        
        
            
            
    def monitor(self, core_id, neuron_id):
        """
        Wrapper function for monitoring I_mem. Accepts core_id and neuron_id from 0 to 255 instead
        of chip_id and neuron_id from 0 to 1023.
        
        Args:
            core_id (int): core ID to be monitored
            neuron_id (int): neuron index within the selected core
        """
        CtxDynapse.dynapse.monitor_neuron(int(core_id / 4), neuron_id + 256*(core_id % 4))
        
        