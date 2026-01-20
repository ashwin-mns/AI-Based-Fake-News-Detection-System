import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import os

class MultimodalDataset(Dataset):
    def __init__(self, metadata_path, max_len=128):
        """
        Args:
            metadata_path (string): Path to the csv file with annotations.
            max_len (int): Maximum length of text sequence.
            
        Expected CSV columns: ['text', 'image_path', 'label']
        """
        self.data = pd.read_csv(metadata_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row['text'])
        image_path = row['image_path']
        label = row['label']

        # Text encoding
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Image encoding
        try:
            # Ensure we can open relative paths if needed, or absolute
            if not os.path.exists(image_path):
                 # Try relative to metadata file derived path if needed, but for now assume full path or correct relative
                 pass
            
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # In a real pipeline, we might skip this sample or log it.
            # For robustness here, we return a zero tensor.
            print(f"Warning: Could not load image at {image_path}. Error: {e}")
            image = torch.zeros((3, 224, 224))

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
