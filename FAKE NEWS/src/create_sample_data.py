import pandas as pd
from PIL import Image
import os


def create_sample_data(data_path='data/train.csv', image_dir='data/images'):
    os.makedirs(image_dir, exist_ok=True)
    
    # Define labels (1: Real, 0: Fake)
    labels = []
    texts = []
    
    # REAL NEWS Samples (Objective, Official, Neutral Tone)
    real_news = [
        "Government announces new tax cuts for small businesses.",
        "Local team wins the championship after 20 years.",
        "New medical breakthrough cures the common cold.", 
        "Tech giant releases new smartphone model today.",
        "World leaders gather for peace summit in Geneva.",
        "WHO reports decline in new COVID-19 cases globally.",
        "Scientists discover water on Mars surface.",
        "Stock market hits record high amid economic growth.",
        "Education minister proposes new curriculum reforms.",
        "City council approves budget for new park construction.",
        "Weather forecast predicts heavy rain this weekend.",
        "Study shows regular exercise improves mental health.",
        "SpaceX successfully launches new satellite into orbit.",
        "Local library hosts annual book fair event.",
        "Researchers develop new solar panel technology."
    ]
    texts.extend(real_news)
    labels.extend([1] * len(real_news))

    # FAKE NEWS Samples (Sensational, Conspiracy, Alarmist)
    fake_news = [
        "Aliens confirmed to be living on the moon base.", 
        "Scientist claims the earth is actually flat.", 
        "Official report states climate change is a hoax.",
        "Internet shuts down globally for 24 hours.",
        "Drinking bleach cures all known viruses instantly.", 
        "Celebrity spotted buying all groceries in the city.",
        "Flying cars to replace all traffic by next year.",
        "Government admits birds are actually surveillance drones.",
        "Secret society controls the weather using 5G towers.",
        "Eating rocks is actually good for your digestion.",
        "Lizard people revealed to be running the government.",
        "Your microwave is spying on you for the FBI.",
        "Chocolate milk comes from brown cows only.",
        "Dinosaurs are still alive and hiding in the jungle.",
        "New law requires everyone to wear purple hats."
    ]
    texts.extend(fake_news)
    labels.extend([0] * len(fake_news))

    # Augment data (Commented out to reduce file count)
    # texts = texts * 5
    # labels = labels * 5

    # Create dummy images (Random noise, no color bias)
    # The model will learn to rely on TEXT more since images are random.
    for i in range(len(texts)):
        # Random visual noise
        img = Image.new('RGB', (224, 224), color=(
            (i * 30) % 255, 
            (i * 50) % 255, 
            (i * 70) % 255
        ))
        img.save(f"{image_dir}/img_{i}.jpg")
    
    # Create CSV
    data = {
        'text': texts,
        'image_path': [f"{image_dir}/img_{i}.jpg" for i in range(len(texts))],
        'label': labels
    }
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print(f"Sample data created at {data_path}")

if __name__ == "__main__":
    create_sample_data()
