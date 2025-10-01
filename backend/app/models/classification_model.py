import torch.nn as nn
import torchvision.models as models 

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)