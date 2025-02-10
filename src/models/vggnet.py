from torch import nn 
from models.parts import DoubleConv2d, DoubleUnConvTranspose2d, TripleConv2d, TripleUnConvTranspose2d
#TODO: Try without BN? - Doesn't learn at all
#TODO: Check sizes? 


class VGGNet(nn.Module):
    def __init__(self, kernel_size):
        super(VGGNet, self).__init__()
        self.conv1 = DoubleConv2d(3, 64, kernel_size=kernel_size) 
        self.conv2 = DoubleConv2d(64, 128, kernel_size=kernel_size)
        self.conv3 = TripleConv2d(128, 256, kernel_size=kernel_size) 
        self.conv4 = TripleConv2d(256, 512, kernel_size=kernel_size) 
        self.conv5 = TripleConv2d(512, 512, kernel_size=kernel_size) 
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2)
       
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 512 * 4 * 4)
        
        self.unconv1 = TripleUnConvTranspose2d(512, 512, kernel_size=kernel_size)
        self.unconv2 = TripleUnConvTranspose2d(512, 256, kernel_size=kernel_size)
        self.unconv3 = TripleUnConvTranspose2d(256, 128, kernel_size=kernel_size)
        self.unconv4 = DoubleUnConvTranspose2d(128, 64, kernel_size=kernel_size)
        self.unconv5 = DoubleUnConvTranspose2d(64, 32, kernel_size=kernel_size)
        self.unconv6 = nn.Conv2d(32, 3, kernel_size=1)

        
    def forward(self, x):
        x, indices1, shape1 = self.conv1(x)
        x, indices2, shape2 = self.conv2(x)
        x, indices3, shape3 = self.conv3(x)
        x, indices4, shape4 = self.conv4(x)
        x, indices5, shape5 = self.conv5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x)) 
        
        x = x.view(-1, 512, 4, 4)  

        x = self.unconv1(x, indices5, shape5)
        x = self.unconv2(x, indices4, shape4)
        x = self.unconv3(x, indices3, shape3)
        x = self.unconv4(x, indices2, shape2)
        x = self.unconv5(x, indices1, shape1)

        return x