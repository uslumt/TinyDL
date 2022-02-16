# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:44:56 2021

@author: uslumt

This module Train the network by desired Number of Epochs 

"""
import numpy as np
from cf import CF_dataset
from PytorchNN import Net
from torch import nn
import torch
import torch.optim as optim
from distiller.scheduler import CompressionScheduler
import distiller
from distiller.policy import QuantizationPolicy
import test_quantizer as t_q

dataset = CF_dataset()

class Train():
    
    def __init__(self, Epoch_Number=0, name = 'None'):
        
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        
        for epoch in range(Epoch_Number):  # loop over the dataset multiple times
        
            running_loss1 = 0.0
            running_loss2 = 0.0
            total_train = 0.0
            correct_train = 0.0
            total_test = 0.0
            correct_test = 0.0
            
            compression_scheduler.on_epoch_begin(epoch)
            
            for i, data in enumerate(dataset.get_trainloader(), 1):
                
                # get the inputs; data is a list of [inputs, labels]
                
                inputs, labels = data
                
                compression_scheduler.on_minibatch_begin(epoch, minibatch_id = data, minibatches_per_epoch = 1)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
            
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                compression_scheduler.before_backward_pass(epoch,
                                                           minibatch_id = data, minibatches_per_epoch = 1, loss = loss)
                
                
                loss.backward( retain_graph=True)
               
                compression_scheduler.before_parameter_optimization(epoch,
                                                                    minibatch_id = data, minibatches_per_epoch = 1, optimizer = optimizer)
                
                optimizer.step()
                
                running_loss1 += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                compression_scheduler.on_minibatch_end(epoch,
                                                       minibatch_id = data, minibatches_per_epoch = 1)
                
            with torch.no_grad():
                for data in dataset.get_testloader():
                    
                    images, labels = data
                    outputs = model(images)
                    
                    loss = criterion(outputs, labels)
                    
                    running_loss2 += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                
            compression_scheduler.on_epoch_end(epoch)
            
            print('\n[%d] Training loss: %.3f & acc: %.3f ' % (epoch + 1,
                            running_loss1/195.3125, correct_train / total_train))
            print('[%d] Test loss: %.3f & acc: %.3f ' % (epoch + 1,
                            running_loss2/20.0, correct_test / total_test), "\n")
            
            train_loss.append(float('%.3f' % (running_loss1/195.3125)))
            train_acc.append(correct_train / total_train)
            test_loss.append(float('%.3f' % (running_loss2/ 20.0)))
            test_acc.append(correct_test / total_test)
            
        print('Training Finished and Saved')
        self.PATH = './cifar_net.pth'
        torch.save(model.state_dict(), self.PATH)
        
        print("\nTrain losses : ", train_loss )
        print("Train Accuracy : ", train_acc)
        print("Test losses : ", test_loss )
        print("Test Accuracy : ", test_acc , "\n")
        
        file = open(name+".npy", "wb")
        np.save(file, [train_loss, train_acc, test_loss, test_acc ])
        file.close
        
        #print(qmodel.weight)
        #print(model.conv1.weight)
        
        self.outputs=outputs
        self.data=data
 
# Training the network for 1 epoch and see the outputs
if __name__ == "__main__":
    
    criterion = nn.CrossEntropyLoss()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    compression_scheduler = CompressionScheduler(model)
    
    
    quantizer = t_q.Quant(model, optimizer)
    policy = QuantizationPolicy(quantizer)
    compression_scheduler.add_policy(policy = policy, starting_epoch = 0, ending_epoch = 100, frequency = 1 )
    
    # print("my Test")
    # print(compression_scheduler.model is model)
    # print("Test end")
    
    Train(100, "DistillerBinary")
    
    # model.
    # validate()
    # save_checkpoint()
    #print(t.outputs)
    
