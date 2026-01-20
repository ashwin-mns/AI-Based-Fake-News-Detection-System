import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        # Use ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the classification head (fc layer)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images):
        features = self.resnet(images)
        # Flatten the output: [Batch, 2048, 1, 1] -> [Batch, 2048]
        return features.view(features.size(0), -1)
