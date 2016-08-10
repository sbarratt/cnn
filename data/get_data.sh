#!/bin/bash
mkdir raw
wget -O raw/train-images.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O raw/train-labels.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O raw/test-images.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O raw/test-labels.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip raw/train-images.gz
gunzip raw/train-labels.gz
gunzip raw/test-images.gz
gunzip raw/test-labels.gz
