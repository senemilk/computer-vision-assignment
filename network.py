import torch
import torch.nn as nn



class detector(nn.Module) :
	def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]
        super(detector, self).__init__()
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
	#RPN
	self.conv_reg = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_cls = nn.Conv2d(feat_in, feature_out*2*anchor, 3)


	def forward(self, x):
        	x_f = self.featureExtract(x)
		return self.conv_reg(x_f),self.conv_cls(x_f)




e
