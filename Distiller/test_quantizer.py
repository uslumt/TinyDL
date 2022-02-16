# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:19:41 2021

@author: uslumt
"""

import yaml
from yaml.loader import SafeLoader
from distiller.quantization.clipped_linear import DorefaQuantizer, WRPNQuantizer,PACTQuantizer

def Quant (Model, opt):
    with open('pytorch_dorefa.yaml', 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    mycls = data['quantizers']['dorefa_quantizer']['class']
    act_bit = data['quantizers']['dorefa_quantizer']['bits_activations']
    weight_bit = data['quantizers']['dorefa_quantizer']['bits_weights']
    
    if mycls == 'DorefaQuantizer' :
        quant = DorefaQuantizer( Model, bits_weights = weight_bit, bits_activations = act_bit, optimizer = opt )
    elif mycls == 'DorefaQuantizer' :
        quant = WRPNQuantizer( Model, bits_weights = weight_bit, bits_activations = act_bit, optimizer = opt )
    else:
        quant = PACTQuantizer( Model, bits_weights = weight_bit, bits_activations = act_bit, optimizer = opt )
    return quant
