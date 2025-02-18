# DCGAN for Face Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate faces from random noise.

## Prerequisites

- Basic knowledge of neural networks, deep learning, CNNs, and GANs.
- Familiarity with PyTorch.

## Installation

Install the following dependencies:

- **PyTorch** (version 2.6.0+cu124)
- **Torchvision** (version 0.21.0+cu124)
- **NumPy** (version 1.23.5)
- **Matplotlib** (version 3.5.2)


Check your installation by running:

```python
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def check_installation():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("torchvision version:", torchvision.__version__)
    print("numpy version:", np.__version__)
    print("matplotlib version:", plt.__version__)

check_installation()
```
## References
- https://www.analyticsvidhya.com/blog/2021/07/deep-convolutional-generative-adversarial-network-dcgan-for-beginners/
- https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


