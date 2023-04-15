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
│   ├── MSE_OBS_1.npy
│   ├── MSE_OBS_2.npy
│   ├── MSE_PDE_1.npy
│   ├── MSE_PDE_2.npy
│   ├── MSE_PDE_3.npy
│   ├── Sigma1.npy
│   ├── Sigma2.npy
│   ├── Sigma3.npy
│   ├── Sigma4.npy
│   └── Sigma5.npy
├── losses.py
├── main.py
├── metrics.py
├── net.py
├── pinn_lb.py
├── test.py
├── test_grad.py
└── viz_history.py

```