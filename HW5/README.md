# PHYS490 - HW5

## Dependencies
main.py:
- json
- argparse
- torch
- sys
- os
- matplotlib
- torch.optim

data_gen.py:
- numpy
- torch.utils.data
- torch

nn_gen.py:
- torch
- torch.nn
- torch.nn.functional

## Hyperparameters

- filepath_data : data location
- learning_rate : rate at which the algorithm learns
- num_epochs : number of epochs
- batch_size : size of data batch for training

```yaml
{
    "filepath_data": "data/even_mnist.csv",
    "learning_rate": 1e-3,
    "num_epochs": 100,
    "batch_size": 1e3
}

## Running `main.py`

To run `main.py`, use

```sh
python3 main.py param/param_file.json --res-path results -n 100 -v 2
```
For help, use
```sh
python3 main.py --help
```
