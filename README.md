<h1 align='center'> Do Conservative PINNs Train Faster In High Dimensions?
    [<a href="https://rfangit.github.io/blog/">Blog</a> </h1>

<p align="center">
<img align="middle" src="./imgs/main.png" width="666" />
</p>

A code repository for experiments that show networks built to to output conservative vector fields train faster than baseline networks on a toy conservative vector flow problem in sufficiently high dimensions. However, the speedup is only modest even at high dimension.

----

## Table of Contents

- **dataset_gen.py** : Functions for generating datasets
- **model.py** : Neural network models (baseline network + conservative network)
- **train.py** : Functions for training neural network models.
- **Experiments (Conservative Fields)**: Code and results for experiments with conservative vector fields in a variety of experimental regimes and network architectures
- **Experiments (Non-conservative Fields)**: Code and results for experiments with non-conservative vector fields to show results obtained are due to the built-in conservative vector field bias.
- **figs**: Code for generating figures.

### Organization of Experiments


### Results
