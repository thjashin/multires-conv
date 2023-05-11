# MultiresConv 

This repository contains the official PyTorch implementation of

### Sequence Modeling with Multiresolution Convolutional Memory (ICML 2023) 
by [Jiaxin Shi](https://jiaxins.io), [Ke Alexander Wang](https://keawang.github.io/), [Emily B. Fox](https://emilybfox.su.domains/) 

**TL;DR:** We introduce a new SOTA convolutional sequence modeling layer that is **simple to implement** (15 lines of PyTorch code using standard convolution and linear operators) and requires at most **O(N log N) time and memory**. 

<img src="results/multires_layer.png" width="600">

The key component of the layer is a multiresolution convolution operation (`MultiresConv`, **left** in the figure) that mimics the computational structure of wavelet-based multiresolution analysis. 
We use it to build a memory ($\mathbf{z}\_n$ in the figure) for long context modeling which captures multiscale trends of the data. 
Our layer is simple (it's linear) and parameter efficient (it uses depthwise convolutions; filters are shared across timescales), making it easy to intergrate with modern architectures such as gated activations, residual blocks, and normalizations. 
