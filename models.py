import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)

class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        feature_extractor = torchvision.models.resnet50(pretrained=P['use_pretrained'])
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
        

        if P['freeze_feature_extractor']:
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True
        self.feature_extractor = feature_extractor
            
        self.avgpool = GlobalAvgPool2d()

        linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
        self.linear_classifier = linear_classifier

    def unfreeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def get_cam(self, x):
        feats = self.feature_extractor(x)
        CAM = F.conv2d(feats, self.linear_classifier.weight.unsqueeze(-1).unsqueeze(-1))
        return CAM

    def foward_linearinit(self, x):
        x = self.linear_classifier(x)
        return x
        
    def forward(self, x):

        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.linear_classifier(pooled_feats)
    
        return logits
