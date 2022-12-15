import torch
import torch.nn as nn

class PickLoss(nn.Module):
    def __init__(self, device) -> None:
        super(PickLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, out, target):
        return self.criterion(
            out["pick"], 
            torch.cat(target["action_instance"], dim=0).to(self.device)
        )

class PlaceLoss(nn.Module):
    def __init__(self, device) -> None:
        super(PlaceLoss, self).__init__()
        self.criterion = torch.nn.SmoothL1Loss()
        self.device = device
    
    def forward(self, out, target):
        assert out['place'].shape[-1] in [3, 7]
        return self.criterion(
            out['place'],
            torch.cat(target['end_pose'], dim=0).to(self.device)[:, out['place'].shape[-1]]
        )

class PickPlaceLoss(nn.Module):
    def __init__(self, alpha, device) -> None:
        super(PickPlaceLoss, self).__init__()
        self.criterion_pick = PickLoss(device)
        self.criterion_place = PlaceLoss(device)
        self.alpha = alpha
        self.device = device

    def forward(self, out, target):
        return self.alpha * self.criterion_pick(out, target) + \
            (1-self.alpha) * self.criterion_place(out, target)
    