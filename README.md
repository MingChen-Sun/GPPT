# GPPT

This repo contains code accompaning the paper, 	[GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks (Mingchen Sun et al., KDD 2022)](https://dl.acm.org/doi/abs/10.1145/3534678.3539249).

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* PyTorch v1.8.+
* DGL v0.7+

### Data
We evaluate our model on eight benchmark datasets, see the usage instructions in `load_graph.py` and `utils.py` respectively.

### Hyperparameters
The hyperparameters settings see `get_args.py`.

### Usage
To run the code, see the usage instructions at the top of `GPPT.py` or `run_all.py`.

