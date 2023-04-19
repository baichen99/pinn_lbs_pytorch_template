## Pinn-based NS Equation Solver

### Description

This project provides a Python implementation of a Pinn-based NS equation solver. The NS equation can be replaced with other equations, as this code serves as a template. Additionally, this code includes the implementation of Self-adaptive loss balanced Physics-informed neural networks as described in Zixue Xiang, Weipeng, Xu Liu, Wen Yao's paper [Self-adaptive loss balanced Physics-informed neural networks](https://doi.org/10.1016/j.neucom.2022.05.015).

The code for the self-adaptive loss balanced Pinn was taken from [this Github repository](https://github.com/xiangzixuebit/LBPINN). The project directory structure is as follows:

```
.
├── README.md
├── data
│   ├── data.npy
│   └── train_data.npy
├── logs
│   ├── history.npy
├── losses.py
├── metrics.py
├── net.py
├── pinn_lb.py
├── test.py
├── test_grad.py
└── viz_history.py

```

## Usage

To use this template, you will need to have Python 3 and PyTorch installed on your system. Clone this repository and use it as a starting point to develop your own Pinn-based solver.

In `losses.py`, the NS equation loss is implemented. You can refer to this code and add your own loss function.

In `pinn_lb.py`, the training process for the self-adaptive loss balanced Pinn is implemented.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.