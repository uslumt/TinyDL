"""
Created on Tue Jun 15 09:44:06 2021

Before running main module please check imported models  in Test_Model and Train_Model modules

@author: uslumt
"""

from Test_Model import Test
from plotting import plot

# Train and test the model by n number of Epoch
n= int(input("Epoch Number = "))

# Enter the Model name that is imported in Train_Model and Test_Model
name = input("Imported Model = ")
t=Test(n, name)
t.accuracy()
plot(n, name)
