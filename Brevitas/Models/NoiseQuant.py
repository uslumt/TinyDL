"""
Created on Wed May  5 15:34:53 2021

Override of the method weight_quant during the training we can add random distribution to weights 

@author: uslumt
"""
from brevitas.nn import QuantConv2d
import numpy as np


class Noise():
    def quant_weight(self):
        
        noise = np.random.normal(0, 0.01) 
        noisy_weight = self.weight + noise
        
        return self.weight_quant(noisy_weight)

class NoiseQuantConv(Noise, QuantConv2d): 
    pass
