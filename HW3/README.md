# PHYS490 - HW3

## Dependencies
main.py:
- json
- argparse
- torch
- sys
- torch.optim
- torch.autograd.Variable
- scipy.integrate.odeint

data.py:
- numpy

nn.py:
- torch
- torch.nn

## Running `main.py`

To run `main.py`, use

```sh
python3 main.py param/param_file.json --x-field y**2 --y-field x**2
```
For help, use
```sh
python3 main.py --help
```
