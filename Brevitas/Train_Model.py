"""
Created on Tue Mar  9 11:44:56 2021


This module Train the network by desired Number of Epochs
Please import the model that you want to train

@author: uslumt
"""
import numpy as np
from cf import CF_dataset
from TernaryNN import Net
from torch import nn
import torch
import torch.optim as optim

model = Net()
dataset=CF_dataset()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class Train():
    
    def __init__(self, Epoch_Number=0, name  =' None'):
        
        
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
            
            for i, data in enumerate(dataset.get_trainloader(), 1):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                
                running_loss1 += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
                
            with torch.no_grad():
                for data in dataset.get_testloader():
                    
                    images, labels = data
                    outputs = model(images)
                    
                    loss = criterion(outputs, labels)
                    
                    running_loss2 += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
                
            print('\n[%d] Training loss: %.3f & acc: %.3f ' % (epoch + 1,
                            running_loss1/195.3125, correct_train / total_train))
            print('[%d] Test loss: %.3f & acc: %.3f ' % (epoch + 1,
                            running_loss2/20.0, correct_test / total_test), "\n")
            
            train_loss.append(float('%.3f' % (running_loss1/195.3125)))
            train_acc.append(correct_train / total_train)
            test_loss.append(float('%.3f' % (running_loss2/ 20.0)))
            test_acc.append(correct_test / total_test)
            
        
        self.PATH = './cifar_net.pth'
        torch.save(model.state_dict(), self.PATH)
        print('Training Finished and Saved')
        
        print("\nTrain losses : ", train_loss )
        print("Train Accuracy : ", train_acc)
        print("Test losses : ", test_loss )
        print("Test Accuracy : ", test_acc , "\n")
                
        file = open(name+".npy", "wb")
        np.save(file, [train_loss, train_acc, test_loss, test_acc ])
        file.close
        print("conv1")
        print(model.qconv1.quant_weight())
        print("conv2")
        print(model.qconv2.quant_weight())
        #print(model.fc1.quant_weight())
        
        self.outputs=outputs
        self.data=data
 
# Training the network for 1 epoch and see the outputs
if __name__ == "__main__":
    
    t=Train(1)
    #print(t.outputs)
    
