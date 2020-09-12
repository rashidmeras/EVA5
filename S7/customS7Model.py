import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # C1 
        self.conv1a = nn.Sequential(nn.Conv2d(3,  32,3, padding=1),            nn.ReLU(), nn.BatchNorm2d(32))
        self.conv1b = nn.Sequential(nn.Conv2d(32,128,3, padding=1, groups=32), nn.ReLU(), nn.BatchNorm2d(128)) 
        # MP
        self.pool1 = nn.Sequential( nn.MaxPool2d(2,2) ) 

        # C2
        self.conv2 = nn.Sequential(nn.Conv2d(128,160,3, padding=1, dilation=2), nn.ReLU(), nn.BatchNorm2d(160)) 
        # MP
        self.pool2 = nn.Sequential(nn.MaxPool2d(2,2) ) 

        # C3
        self.conv3 = nn.Sequential(nn.Conv2d(160,320,3, padding=2), nn.ReLU(), nn.BatchNorm2d(320) )
        # MP
        self.pool3 = nn.Sequential(nn.MaxPool2d(2,2) )

        # C4
        self.conv4 = nn.Sequential(nn.Conv2d(320,96,3, padding=1), nn.ReLU(), nn.BatchNorm2d(96))

        # Layer5:
        self.conv5 = nn.Sequential(nn.Conv2d(96, 10, 1))  

        # GAP Layer
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4)) # output_size = 1     


    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
        
