"""
Created on Tue Jun  8 14:41:40 2021

In this module we implemented Binary quantization,
during the training execution of weigths we shoud be able see 2 level outputs
 
@author: uslumt
"""
from torch.nn import Module
from brevitas.nn import  QuantIdentity, QuantConv2d, QuantReLU, QuantLinear, QuantMaxPool2d, BatchNorm2dToQuantScaleBias
from brevitas.inject.enum import QuantType

class Net(Module):
    
    def __init__(self):
    
        super(Net, self).__init__()
        
        self.quant_inp = QuantIdentity(bit_width=2)
        self.qconv1 = QuantConv2d(3, 6, 5, weight_quant_type = QuantType.BINARY, weight_bit_width = 1 ) 
        self.pool1 = QuantMaxPool2d(2, 2)                               
        self.BN1 = BatchNorm2dToQuantScaleBias(6, affine = False)           
        self.relu1 = QuantReLU(bit_width = 2)                 
        self.qconv2 = QuantConv2d(6, 16, 5, weight_quant_type = QuantType.BINARY, weight_bit_width = 1 )
        self.BN2 =  BatchNorm2dToQuantScaleBias(16, affine = False)
        self.pool2 = QuantMaxPool2d(2, 2)
        self.relu2 = QuantReLU(bit_width = 2)
        self.fc1 = QuantLinear(400, 100, bias = True, weight_quant_type = QuantType.BINARY, weight_bit_width = 1 ) 
        self.fc2 = QuantLinear(100, 50, bias = True, weight_quant_type = QuantType.BINARY, weight_bit_width = 1 )
        self.fc3 = QuantLinear(50, 10, bias = True,  weight_quant_type =QuantType.BINARY, weight_bit_width=  1 )
        self.relu3 = QuantReLU(bit_width = 2)
        self.relu4 = QuantReLU(bit_width = 2)
        
    def forward(self, x):
        
        x = self.quant_inp(x)
        x = self.BN1(self.pool1(self.qconv1(x)))
        x = self.relu1(x)
        x = self.BN2(self.pool2(self.qconv2(x)))
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        
        return x
