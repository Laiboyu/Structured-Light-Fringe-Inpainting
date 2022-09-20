import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:                                                                            
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
    
class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
    
    def forward(self, input, target):
        
        # input = torch.sigmoid(input)
        
        N = target.size(0)
        smooth = 1
        
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        
        intersection = torch.abs(input_flat * target_flat)
        
        dice_value = 2 * (intersection.sum(1) + smooth) / (torch.abs(input_flat.sum(1)) + torch.abs(target_flat.sum(1)) + smooth)
        dice_value = dice_value.sum() / N     
        # intersection = input_flat * target_flat
        
        # dice_value = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        # dice_value = dice_value.sum() / N     
        
        return dice_value
    
class MIoU(nn.Module):
    def __init__(self):
        super(MIoU, self).__init__()
        
    def forward(self, input, target):
        
        # input = torch.sigmoid(input)
        
        N = target.size(0)
        smooth = 1
        
        input_flat  = input.view(N, -1)
        target_flat = target.view(N, -1)
        
        intersection = input_flat * target_flat
        overlap = (input_flat + target_flat) - intersection
        
        mean_iou_value = (intersection.sum(1) + smooth) / (overlap.sum(1) + smooth)
        mean_iou_value = mean_iou_value.sum() / N
        
        return mean_iou_value