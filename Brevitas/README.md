In this directory we used [**Brevitas**](https://github.com/Xilinx/brevitas) tool for Quantization Aware Training (QAT) within [_PyTorch_](https://pytorch.org/) framework on [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset .

Our work can be improved and more compact by some config. files but we wanted to see what results we can get from this tool and how we can use the tool for future projects.

## Run Code:

There are several small Models (e.g. BinaryNN, TernaryNN) those we created by using Brevitas nn modules.
To run the whole embedded modules we use main.py, before that make sure you check what Models you imported in Test_Model.py and Train_Model.py

During the operation of main.py Program will ask you two inputs :
**n** = Number of epochs (integer)
**name** = Model name (Imported one Model name in Test_Model.py and Train_Model.py ) It`s a string e.g. TernaryNN

During the trainig, program executes some results such as Accuracy and Loss of the repective epoch, to make sure we printed quantized weights of 1.st conv. layer of quantized Model whether our implementation is correct so for example in TernaryNN we should expect 3 levels of weighs [+value, 0, -value ...] 
Note : Use of PytorchNN will reise an error because Pytorch module doesn`t have quantization method.

## Outputs: In console we should see the statistics and two plots that show test and train Accuracy & loss vs epoch number.
We also saved these statistics in a .npy file which has the same Model name we imported


## F.A.Q.                
**Q** : Is there any paper that the quantization idea is taken from ?
**A**: See [Brevitas Paper](https://arxiv.org/pdf/1907.00593.pdf) and also take a look this [Paper](https://arxiv.org/abs/1903.08066v2)
**Q**: Why should we use Brevitas?
**A**: It`s a very flexible tool for Quantization Aware Training, comparison to Tensorflow Lite quantization Brevitas supports lower level quantization such as Ternary, Binary.
Quite active community and support for bug fix and also the tool is still in development process.

**Q**: How hard to install Brevitas and get started with it?
**A**: Brevitas doesn`t require very specific version of PyTorch and Python, visit Brevitas repo for more information

**Q**: Can I use Brevitas modules with PyTorch modules?
**A**: Yes, make sure you pass the parameter `return_quant_tensor = True` that integrates Brevitas tensors to PyTorch tensors.

**Some Motivating Results**:
 With 100 Epochs

|| PyTorch | TernaryNN | BinaryNN |
| ------ | ------ |------ |------ |
| Train Acuracy | 0.98 | 0.62 |0.55 |
| Test Acuracy | 0.57 | 0.56 |0.51 |




## Authors:
Maen Mallah Research Associate @Fraunhofer IIS,
Muhammet Uslu Student Assistant @Fraunhofer IIS
