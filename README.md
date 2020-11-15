# BNN_psum_quantized.pytorch
Binarized Neural Network (BNN) Partial Sum Quantization pytorch
This is the pytorch version for quantizing partial sums in BNNs
Link to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

The code is based on https://github.com/itayhubara/BinaryNet.pytorch
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10
