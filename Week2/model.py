import torch 
nn = torch.nn
F = nn.functional
class Net(nn.Module):
    def __init__(self): 
         super(Net, self).__init__()
         self.fc1 = nn.Linear(784,196,bias=True)
         self.fc2 = nn.Linear(196, 50,bias=True)
         self.relu = nn.ReLU()
         self.fc3 = nn.Linear(50, 10,bias=True)
         
    def forward(self,x):
         x=(self.fc1(x))
         x = self.relu(x)
         x=(self.fc2(x))
         x = self.relu(x)
         x=(self.fc3(x))
         return F.softmax(x,dim=-1)
