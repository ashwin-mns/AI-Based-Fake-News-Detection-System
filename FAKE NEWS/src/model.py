import torch
import torch.nn as nn
from src.text_encoder import TextEncoder
from src.image_encoder import ImageEncoder

class FakeNewsModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FakeNewsModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        
        # Dimensions: BERT hidden size (768) + ResNet50 output (2048)
        input_dim = 768 + 2048
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)
        
        # Concatenate features
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # Classification
        output = self.fusion(combined_features)
        return output
