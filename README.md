# AI-Based Fake News Detection System (Multimodal)

This project implements a comprehensive **Multimodal Fake News Detection System** that checks the authenticity of news articles using both **text** (BERT) and **images** (ResNet). It leverages deep learning techniques to analyze semantic integrity and visual features to classify news as **Real** or **Fake**.

## 🚀 Features

*   **Multimodal Analysis**: Combines text and image features for higher accuracy.
*   **BERT-based Text Model**: Uses a pre-trained `bert-base-uncased` model for state-of-the-art text understanding.
*   **ResNet-based Image Model**: Utilizes `ResNet50` (or similar CNN architecture) to extract visual features.
*   **User-Friendly App**: Interactive Streamlit web application for real-time detection.
*   **Custom Dataset Support**: Easy-to-use CSV format for training on your own datasets.

## 🧠 Why Deep Learning? (vs. Traditional ML)

This project generally outperforms traditional Machine Learning algorithms like KNN, Random Forest, or Decision Trees for this specific task due to the complexity of the data involved.

| Feature | Traditional ML (RF, KNN) | Our Approach (BERT + ResNet) |
| :--- | :--- | :--- |
| **Understanding** | Checks for specific keywords | Understands meaning and context |
| **Images** | Very poor / Requires manual feature extraction | State-of-the-art Computer Vision |
| **Complexity** | Simple, fast to train | Complex, high accuracy, GPU accelerated |

### Key Advantages:
*   **Text (BERT)**: Understands the context, grammar, and meaning of the full sentence. It treats language like a human does, capturing sarcasm, nuance, and semantic relationships.
*   **Images (ResNet)**: Automatically learns to identify objects, scenes, and complex patterns directly from raw pixels.
*   **Fusion**: Our model actively learns the *relationship* between the text and image, whereas traditional methods often treat them as separate, independent features.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Deep Learning**: PyTorch
*   **NLP**: Hugging Face Transformers (BERT)
*   **Computer Vision**: Torchvision (ResNet)
*   **Web Framework**: Streamlit
*   **Data Handling**: Pandas, NumPy

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment** (Optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset Preparation

To train the model, you need a CSV file (default: `data/train.csv`) with the following columns:

| Column | Description |
| :--- | :--- |
| `text` | The headline or content of the news article. |
| `image_path` | Relative or absolute path to the associated image file. |
| `label` | `1` for **Real News**, `0` for **Fake News**. |

**Example structure:**
```csv
text,image_path,label
"Breaking news about AI",data/images/img1.jpg,1
"Aliens landed in backyard",data/images/alien.jpg,0
```

*Tip: You can generate dummy sample data for testing by running:*
```bash
python src/create_sample_data.py
```

## 🖥️ Usage

### 1. Training the Model
To train the model on your dataset, run:
```bash
python -m src.train
```
*   This will download the pre-trained BERT and ResNet models on the first run.
*   The trained model weights will be saved as `fake_news_model.pth`.

### 2. Running the Application
Launch the web interface to test the model:
```bash
streamlit run app.py
```
*   Upload an image and enter text to get a real-time prediction.

## 📂 Project Structure

```
├── data/                   # Dataset directory
├── src/                    # Source code
│   ├── dataset.py          # Data loading and processing
│   ├── model.py            # Multimodal model architecture
│   ├── train.py            # Training loop
│   ├── text_encoder.py     # BERT wrapper
│   └── image_encoder.py    # ResNet wrapper
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🤝 Contributing
Contributions are welcome! Please Open an issue or submit a pull request for any improvements.

## 📜 License
[MIT License](LICENSE)

<img width="1917" height="863" alt="Screenshot 2026-01-29 183402" src="https://github.com/user-attachments/assets/15afbcba-5be5-4552-bba2-34b5961d4d53" />

<img width="1906" height="865" alt="Screenshot 2026-01-29 183539" src="https://github.com/user-attachments/assets/b5836c18-a39e-41fb-8f68-d057fb8512d1" />

<img width="1906" height="863" alt="Screenshot 2026-01-29 183707" src="https://github.com/user-attachments/assets/11bd118f-68ff-4f76-8a25-b3ad717a9927" />

<img width="1903" height="849" alt="Screenshot 2026-01-29 183733" src="https://github.com/user-attachments/assets/e0840d79-4d91-4103-8f55-39fd6253e4d7" />


