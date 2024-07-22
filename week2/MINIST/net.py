import torch
import torch.nn as nn
import torch.optim as optim


class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=120,out_features=84)
        self.linear2 = nn.Linear(in_features=84,out_features=10)
        self.softmax = nn.Softmax()

    def forward(self,x):
        
        x = self.sigmoid((self.conv1(x)))
        x = self.pool1(x)
        
        x = self.sigmoid((self.conv2(x)))
        x = self.pool2(x)
        
        x = self.conv3(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        
        return x

if __name__ == "__main__":
    x = torch.rand([1,1,28,28])
    model = MyLeNet()
    y = model(x)
    print(model)


