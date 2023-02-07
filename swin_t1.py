from cProfile import label
import torch
import torch.nn as nn
from torchvision import models
from dataset2 import *
# from dataset import *

class swin_t1(nn.Module):
    def __init__(self, num_class, freeze):
        super(swin_t1, self).__init__()
        self.backbone = nn.Sequential(*list(models.swin_t(weights=models.Swin_T_Weights.DEFAULT).children())[:-4])
        if freeze == True:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Conv2d(768, num_class, 1)
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.avg_pool(torch.permute(out, [0, 3, 1, 2]))
        out = self.fc(out)
        
        return out.squeeze()