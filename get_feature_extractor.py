import sys
sys.path.append('/home/malick/Bureau/DaSiamRPN/code/')

import torch.nn as nn
import torch.nn.functional as F
import torch


class SiamRPN_fe(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]
        super(SiamRPN_fe, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            )
        
        self.cfg = {}

    def forward(self, x):
        x_f = self.featureExtract(x)
        return x_f



class SiamRPNvot_fe(SiamRPN_fe):
    def __init__(self):
        super(SiamRPNvot_fe, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr':0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 600, 'adaptive': False} # 0.355




# load net
from net import SiamRPNvot
from os.path import realpath, dirname, join
import os
net = SiamRPNvot()
net.load_state_dict(torch.load('/home/malick/Bureau/DaSiamRPN/code/SiamRPNVOT.model'))
net.eval()

#save and load model
fe = SiamRPNvot_fe()
fe_dict = fe.state_dict()
net_dict = net.state_dict() 
new_dict = {k: v for k, v in net_dict.items() if k in fe_dict }
fe_dict.update(new_dict)
fe.load_state_dict(fe_dict)

#freeze parameters of feature extractor to avoid computing gradient 

for param in fe.parameters():
    param.requires_grad = False

#save the feature extractor
torch.save(fe.state_dict(), "feature_extract.pth")



