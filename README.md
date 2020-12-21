# Regularized BNN & Modified Backward STE 
This repository contains the code for [(arxiv link)](https://arxiv.org/pdf/1812.11800.pdf)

Specifically, it contains my implementation of regularized binary network trianing with scaling factors. Models can be learned with different activations and backward STEs. Refer to the paper for more details.

## Dependencies

This project in a conda virtual environment on Ubuntu 16.04 with CUDA 10.0. Dependencies:
* [pytorch with torchvision](http://pytorch.org/). Note that you need at least a fairly recent version of PyTorch (e.g., 1.0). 
* [tensorflow with tensorboard](https://www.tensorflow.org/install/) (This is not critical, however, and you can comment out all references to TF and TB) 

## Running the experiments
Experiment parameters are set in the configs/*.json files. 
There are a number of arguments available, including 
* reg_type: specify the type of regularization to use (e.g. l1, l2)
* reg_scale: flag for introduce scaling factors in conv layers
* reg_lr: learning rate schedule for regularization
* activation: backward ste to be used (SWISH, STE, BIREAL, etc.)

refer to ./configs/config_bnn.json for sample config file.


### Running an experiment 

Sample run of the code with L1 regularizer, HardTanh activation, STE backward, 
```
python train.py -c ./configs/config_bnn.json
```

## Acknowledgments

This repository includes code from:
* The [PyTorch AlexNet implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py) adapted based on code from the [DoReFa implementation](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py).
* The 8-layer convnet from the [DoReFa repo](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/svhn-digit-dorefa.py).
* BNN modules from https://github.com/itayhubara/BinaryNet.pytorch
* TensorboardLogger from [this gist](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) 


