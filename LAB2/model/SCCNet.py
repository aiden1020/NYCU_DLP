import torch
import torch.nn as nn
import torch.nn.functional as F

class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, Nc=22, C=22, Nt=1, dropoutRate=0.7):
        super(SCCNet, self).__init__()

        self.conv1 = nn.Conv2d(1, Nu, (C, Nt), padding=0)
        self.bn1 = nn.BatchNorm2d(Nu)
        
        padding_width = 12 // 2
        self.conv2 = nn.Conv2d(1, Nc, (Nu, 12), padding=(0, padding_width))  
        self.bn2 = nn.BatchNorm2d(Nc)
        self.square = SquareLayer()
        self.dropout = nn.Dropout(dropoutRate)
        
        self.avg_pool = nn.AvgPool2d((1, 62), stride=(1, 12))
        
        output_width = ((timeSample - Nt + 1 - 62) // 12 + 1)
        in_features = Nc * output_width
        self.fc = nn.Linear(in_features, numClasses)

    def forward(self, x):
        x = x.unsqueeze(1) 
        
        x = self.conv1(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 1, 3) 
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.square(x)

        x = self.dropout(x)
        
        x = self.avg_pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = F.softmax(x, dim=1)
        return x   
