# currency_detection_app/models.py
import torch
import torch.nn as nn
import torchvision.models as models

class FusionModelImproved(nn.Module):
    def __init__(self, num_classes):
        super(FusionModelImproved, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.vgg.classifier = nn.Identity()
        self.capsule = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.vgg(x)
        output = self.capsule(features)
        return output
