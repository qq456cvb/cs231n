# CS231n: Convolutional Neural Networks for Visual Recognition — Assignment Solutions

Complete solutions for the assignments of Stanford's [CS231n](https://cs231n.stanford.edu/) (2016 edition), implemented in NumPy from scratch. All coding parts are completed; a few inline written questions are left unanswered.

## Contents

### Assignment 1 — Image Classification Basics
- `knn.ipynb` — k-Nearest Neighbor classifier with vectorized distance computation
- `svm.ipynb` — Multiclass SVM loss, analytic vs. numerical gradients, SGD
- `softmax.ipynb` — Softmax classifier with cross-entropy loss
- `two_layer_net.ipynb` — Two-layer fully-connected network with backpropagation
- `features.ipynb` — Classification on HOG/color-histogram features

### Assignment 2 — Neural Networks and Convolution
- `FullyConnectedNets.ipynb` — Modular layer design, arbitrary-depth networks, SGD variants (momentum, RMSProp, Adam)
- `BatchNormalization.ipynb` — Batch normalization forward/backward
- `Dropout.ipynb` — Dropout regularization
- `ConvolutionalNetworks.ipynb` — Convolution/pooling layers and a working CNN

### Assignment 3 — Recurrent Networks and Visualization
- `RNN_Captioning.ipynb` — Vanilla RNN for image captioning on MS-COCO
- `LSTM_Captioning.ipynb` — LSTM-based captioning
- `ImageGradients.ipynb` — Saliency maps and fooling images
- `ImageGeneration.ipynb` — Class visualization, feature inversion, DeepDream

## Usage

Each assignment folder is self-contained with its own `requirements.txt`. Datasets (CIFAR-10, MS-COCO features) are downloaded by the scripts provided in each assignment's `cs231n/datasets/` directory.

Note: this code targets the 2016 course environment (Python 2.7 era). Expect minor adjustments (e.g., `print` statements, old NumPy APIs) when running on a modern stack.

## Acknowledgements

Assignment scaffolding by the CS231n course staff. See the [course notes](https://cs231n.github.io/) for the accompanying material.
