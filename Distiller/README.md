In this repo we quantize our small model with [**Distiller**](https://github.com/IntelLabs/distiller) IntelLabs tool.
Please take a look at this [**Wiki Page**](https://intellabs.github.io/distiller/index.html) to understand how quantization is done and more background information from the development team.
Basically quantization methods are embedded into training procedure by calling compression_scheduler; that`s why we can use any popular modules written in PyTorch framework, but please make sure prepare the model for quantizatiion operation. (See Wiki Page).
Distiller github page does not have so many examples to understand the code and how to implement quatization, sake of convenience they use Yaml file to schedule the quantization and determine the quantization policy.
In our implementation we simulate their file organization and trainig procedure.



**Results** from Distiller end of 100 epochs:

| | 2 bits Quant | 3 Bits Quant |
| ------| ------ | ------ |
| Train Acuracy | 0.52 | 0.64 |
| Test Acuracy | 0.49 | 0.56 |

