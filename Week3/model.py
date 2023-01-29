import torch.nn as nn
import torch
class Net(nn.Module):
    def __init__(self): 
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,padding=2)   
          self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,padding=2)
          self.maxpool = nn.MaxPool2d(2,stride=2)
          self.fc1 = nn.Linear(16*8*8, 120, bias=True)
          self.fc2 = nn.Linear(120,84, bias=True)
          self.fc3 = nn.Linear(84,10, bias=True)
          self.relu = nn.ReLU()
          self.bn1 = nn.BatchNorm1d(120)
          self.bn2 = nn.BatchNorm1d(84)
          self.softmx = nn.Softmax(dim=-1)
          
    def forward(self,x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.bn1(self.fc1(x)))      
        x = self.relu(self.bn2(self.fc2(x)))  
        x = self.softmx(self.fc3(x))
        return x
