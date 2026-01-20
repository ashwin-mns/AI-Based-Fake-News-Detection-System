import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import MultimodalDataset
from src.model import FakeNewsModel
import os

def train(epochs=10, batch_size=4, lr=2e-5, data_path='data/train.csv'):
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please create it or run 'python src/create_sample_data.py'")
        return

    print("Initializing Training...")
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("Loading Data...")
    try:
        dataset = MultimodalDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Model
    print("Loading Model (BERT + ResNet)...")
    model = FakeNewsModel(num_classes=2).to(device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Starting Training Loop...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 2 == 0:
                print(f"  Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Save
    print("Saving Model...")
    torch.save(model.state_dict(), 'fake_news_model.pth')
    print("Model saved to fake_news_model.pth")

if __name__ == "__main__":
    train()
