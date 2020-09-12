import torch
import torch.nn as nn
import torch.nn.functional as F


# create the instance of the class
class CustomS6Model(nn.Module):

    '''
    This is a CustomS6Model model basically built for training with MNIST dataset.
    
    Following is the specification of this netwrork:
    It has 7 convolution layers, a max-pool layer is placed between layer2 and layer3 and
    a GAP layer is placed at the end.

    Total params: 9,850
    Trainable params: 9,850
    Non-trainable params: 0
    '''

    #construct the class
    def __init__(self, gbn_en):

        #call parent class and initilize
        super(Net, self).__init__(gbn_en)
        self.gbn_en = gbn_en

        print("Create the instance of the Net class with GBN = {}".format(gbn_en))

        # Layer1:
        if gbn_en:
            self.conv1 = nn.Sequential (nn.Conv2d(1, 32, 3, padding=1),GhostBatchNorm(32,2),nn.ReLU())
        else:  
            self.conv1 = nn.Sequential (nn.Conv2d(1, 32, 3, padding=1),nn.BatchNorm2d(32),nn.ReLU())

        # Layer2: 
        if gbn_en:
            self.conv2 = nn.Sequential (nn.Conv2d(32, 16, 3),GhostBatchNorm(16,2),nn.ReLU())
        else:             
            self.conv2 = nn.Sequential (nn.Conv2d(32, 16, 3),nn.BatchNorm2d(16),nn.ReLU())

        # Max-Pooling layer 
        self.pool1 = nn.MaxPool2d(2, 2)        

        # Layer3:
        if gbn_en:
            self.conv3 = nn.Sequential (nn.Conv2d(16, 8, 1),GhostBatchNorm(8,2),nn.ReLU(),nn.Dropout(0.1))
        else:               
            self.conv3 = nn.Sequential (nn.Conv2d(16, 8, 1),nn.BatchNorm2d(8),nn.ReLU(),nn.Dropout(0.1))

        # Layer4:
        if gbn_en:
            self.conv4 = nn.Sequential (nn.Conv2d(8, 16, 3),GhostBatchNorm(16,2),nn.ReLU())
        else:               
            self.conv4 = nn.Sequential (nn.Conv2d(8, 16, 3),nn.BatchNorm2d(16),nn.ReLU())
            
        # Layer5:
        if gbn_en:
            self.conv5 = nn.Sequential (nn.Conv2d(16, 14, 3),GhostBatchNorm(14,2),nn.ReLU())
        else:          
            self.conv5 = nn.Sequential (nn.Conv2d(16, 14, 3),nn.BatchNorm2d(14),nn.ReLU())

        # Layer6:
        if gbn_en:
            self.conv6 = nn.Sequential (nn.Conv2d(14, 10, 3),GhostBatchNorm(10,2),nn.ReLU())
        else:           
            self.conv6 = nn.Sequential (nn.Conv2d(14, 10, 3),nn.BatchNorm2d(10),nn.ReLU())

        # Layer7:
        self.conv7 = nn.Sequential (nn.Conv2d(10, 10, 1))            

        # GAP Layer
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=7)) # output_size = 1     

    # defines the strcuture of the class
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
