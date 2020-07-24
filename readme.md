# Neural Turing Machine

Neural Turing Machine is a neural network (LSTM Cell) coupled with an external memory with which the neural network interacts (read and write) using attention processes. NTMs can be used as a model-based technique for meta-learning.

## Requirements
* PyTorch
* Torchvision
* Numpy
* Tqdm

## Usage
1. Run the [train.py](train.py) script for training the model with default settings. For options, `python train.py -h`
2. Run the [Predict.ipynb](Predict.ipynb) notebook for making predictions and visualizing the results.
3. Alternatively, run this repository directly in colab <a href="https://colab.research.google.com/drive/1YoM9QzSGdvlK7MK7BrHBqdgpXERiNBac?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Credits
* Neural Turing Machines https://arxiv.org/abs/1410.5401
* Code is based on https://github.com/vlgiitr/ntm-pytorch
* https://www.niklasschmidinger.com/posts/2019-12-25-neural-turing-machines/
* Implementing Neural Turing Machines https://arxiv.org/abs/1807.08518
