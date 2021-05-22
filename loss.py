import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tqdm import tqdm

from configure import Config
config = Config()

torch.manual_seed(config.seed_value) # cpu  vars
torch.cuda.manual_seed(config.seed_value)
torch.cuda.manual_seed_all(config.seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False

class WBCELoss(nn.Module):
    def __init__(self, weight= None, size_average=True):
        super(WBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
#        num = targets.size(0)  # Number of batches

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        N = len(targets)
        wf = targets.sum() / N
        wb = (1-targets).sum() / N
        loss = targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)

        return torch.neg(torch.mean(loss) ) #/ num)

class TopKLoss(nn.Module):
    def __init__(self, weight= None, size_average=True):
        super(TopKLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
#        num = targets.size(0)  # Number of batches

        thr = 0.7

        tmp = inputs
        pinp = inputs
        torch.where(tmp<=thr, inputs, torch.ones(inputs.shape).to('cuda'))
        torch.where(tmp>thr, inputs, torch.zeros(inputs.shape).to('cuda'))
        loss = targets * inputs * torch.log(pinp) + ((1 - targets) * (1 - inputs) * torch.log(1 - pinp))

        return torch.neg(torch.mean(loss) ) #/ num)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
#        num = targets.size(0)  # Number of batches

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice.sum() #/ num

class GDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
#        num = targets.size(0)  # Number of batches

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        N = len(targets)
        wf = targets.sum() / N
        wb = (1-targets).sum() / N
        w = wf*targets + wb*(1-targets)

        intersection = (w * inputs * targets).sum() 
        dice = (2.*intersection + smooth)/((w * inputs).sum() + (w * targets).sum() + smooth)  
        
        return 1 - dice.sum() #/ num

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
#        num = targets.size(0)  # Number of batches

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = 0.2 * BCE + 0.8 * dice_loss
        
        return Dice_BCE

class ELLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ELLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
#        num = targets.size(0)  # Number of batches

        gama = 2

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = functional.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = 0.2 * torch.pow(-torch.log(BCE), gama) + 0.8 * torch.pow(-torch.log(dice_loss), gama)
        
        return Dice_BCE
