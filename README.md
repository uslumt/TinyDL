# TinyDL
## Introduction
Quantization is the process of transforming deep learning models to use parameters and computations at a lower precision. 
Traditionally, DNN training and inference have relied on the [IEEE single-precision floating-point format](https://ieeexplore.ieee.org/document/4610935), 
using 32 bits to represent the floating-point model weights and activation tensors.
This compute budget may be acceptable at training as most DNNs are trained in data centers or GPUs. However, during deployment, 
these models are most often required to run on devices with much smaller computing resources and lower power budgets at the edge. 
Running a DNN inference using the full 32-bit representation is not practical for real-time analysis given the compute, memory, and power constraints of the edge.

## Quantization Aware Training 
