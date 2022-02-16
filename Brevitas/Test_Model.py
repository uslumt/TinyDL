"""
Created on Wed Mar  3 10:26:36 2021

Please import the model that you want to test 

@author: uslumt
"""

from cf import CF_dataset
from TernaryNN import Net
from Train_Model import Train
import torch

class Test():
    
    dataset=CF_dataset()
    
    dataiter = iter(dataset.get_testloader())

    images, labels = dataiter.next()
    
    def __init__(self, num, name):
            
        self.num=num
        self.train=Train(num, name)
          
    @classmethod
    def truth(cls):
        
        print('GroundTruth: ', ' '.join('%5s' % cls.dataset.classes[cls.labels[j]] for j in range(4)))
   
    def predict(self):
        
        outputs=self.train.outputs
        
        _, predicted = torch.max(outputs, 1)
        
        print('Predicted: ', ' '.join('%5s' % Test.dataset.classes[predicted[j]]
                                      for j in range(4)))
    
    def accuracy(self):
        model = Net()
        model.load_state_dict(torch.load(self.train.PATH))
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in Test.dataset.get_testloader():
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
        
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in Test.dataset.get_testloader():
                images, labels = data
                outputs =model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                Test.dataset.classes[i], 100 * class_correct[i] / class_total[i]))
            

if __name__ == "__main__":

    t=Test(1)
    t.truth()
    t.predict()
    t.accuracy()
