import torch
from . import meta_model
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy
class incetionV3_loss(nn.Module):
    def __init__(self,smoothing):
        super().__init__()
        self.loss_cls=nn.MSELoss()
        self.loss_avg=nn.MSELoss()
    def forward(self,inputs,target):
        if isinstance(inputs,tuple):
            out,avg=inputs
            return self.loss_cls(out.squeeze(),target)+self.loss_avg(avg.squeeze(),target)
        return self.loss_cls(inputs,target)
def build_model(configs,num_classes):
    
    model= getattr(meta_model,f"build_{configs['name']}")(configs,num_classes=num_classes)
    return model