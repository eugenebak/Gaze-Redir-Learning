import torch
import torch.nn as nn
from torchvision import models

class Eyeresnet(nn.Module):
    def __init__(self):
        super(Eyeresnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=48, bias=True)
        self.fc2 = nn.Linear(in_features=48, out_features=2, bias=True)
        
    def forward(self, x, use_feature=False):
        feat = self.resnet18(x)
        out = self.fc2(feat)
        
        if use_feature:
            return out, feat
        else:
            return out

class EyeFeatureExtractor(nn.Module):
    def __init__(self):
        super(EyeFeatureExtractor, self).__init__()
        self.FC1 = nn.Linear(1000, 512, bias=True)
        self.FC2 = nn.Linear(512, 256, bias=True)
        self.FC3 = nn.Linear(256, 48, bias=True)
        self.act = nn.LeakyReLU()
        nn.init.kaiming_normal_(self.FC1.weight.data)
        nn.init.constant_(self.FC1.bias.data, val=0)
        nn.init.kaiming_normal_(self.FC2.weight.data)
        nn.init.constant_(self.FC2.bias.data, val=0)
        nn.init.kaiming_normal_(self.FC3.weight.data)
        nn.init.constant_(self.FC3.bias.data, val=0)
        
    def forward(self, x):
        x = self.act(self.FC1(x))
        x = self.act(self.FC2(x))
        x = self.FC3(x)
        return x

if __name__=="__main__":
    model = Eyeresnet()
    
    input = torch.randn(15, 3, 256, 64)
    output = model(input)
    
    print(input.shape)
    print(output.shape)
    
        