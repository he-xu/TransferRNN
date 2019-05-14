# TransferRNN
Workgroup Project at CapoCaccia 2019

TransferRNN framework implements an RNN running on Dynapse chip using CortexControl Python API

## Installation

1. Download the nightly build of CortexControl from http://ai-ctx.gitlab.io/ctxctl/primer.html#downloads

2. Clone TransferRNN inside the main CortexControl folder
```
clone https://github.com/he-xu/TransferRNN.git
```
3. Prepare the `x_projection_train.pkl` and `state_train.pkl` files (*by default the script will look for them in `TranserRNN/data/`*)

## Running

The framework can run in two modes:
- ctxctl : inside CortexControl (faster but trickier to setup)
- rpyc : in a remote python console using RPyC connection (more stable but slower because of data transfering between python threads)

### `ctxctl` mode

1. Start Cortexcontrol: `./cortexcontrol`
2. In the CortexControl console run `import TransferRNN.run_rnn_in_ctxctl`

**Note:** Most likely, you will get an error while importing numpy. To fix that, you will have to make sure you have `numpy` version `1.15.4` (with `pip install numpy=1.15.4`) in your installation of python and then append the path to `site-packages` of this python installation inside the `TransferRNN/run_rnn_in_ctxctl.py` script on **line 12**.

### `rpyc` mode
0. Install `RPyC` with `pip install rpyc`

1. Start CortexControl: `./cortexcontrol`
2. In the CortexControl console run `PyCtxUtils; PyCtxUtils.start_rpyc_server()`
3. Run `run_rnn_with_rpyc.py` in a python console of your choice (tested in Spyder)

**Note:** Expect the spike counting phase to be slow. Optimization of this phase is the next framework development step.
