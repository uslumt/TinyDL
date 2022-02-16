# TinyDL
This a side project to explore different techniques and tools which are dedicated to Deep Learning neural networks acceration.

## Quantization Aware Training 
Quantization is the process of transforming deep learning models to use parameters and computations at a lower precision. 
Traditionally, DNN training and inference have relied on the [IEEE single-precision floating-point format](https://ieeexplore.ieee.org/document/4610935), 
using 32 bits to represent the floating-point model weights and activation tensors.
This compute budget may be acceptable at training as most DNNs are trained in data centers or GPUs. However, during deployment, 
these models are most often required to run on devices with much smaller computing resources and lower power budgets at the edge. 
Running a DNN inference using the full 32-bit representation is not practical for real-time analysis given the compute, memory, and power constraints of the edge.  
As opposed to computing scale factors to activation tensors after the DNN is trained (also called hard quantization), the quantization error is considered when training the model. The training graph is modified to simulate the lower precision behavior in the forward pass of the training process. This introduces the quantization errors as part of the training loss, which the optimizer tries to minimize during the training. Thus, QAT helps in modeling the quantization errors during training and mitigates its effects on the accuracy of the model at deployment.
<p align="center">
  <img src="https://github.com/uslumt/TinyDL/blob/main/Figures/Figure_unquantized.png" width="200"  lenght="200"  />
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/uslumt/TinyDL/blob/main/Figures/Figure_quantized.png" width="200"  lenght="200" /> 
</p>

