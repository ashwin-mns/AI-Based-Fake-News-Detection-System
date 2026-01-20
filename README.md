# AI-Based Fake News Detection System (Multimodal)

This project implements a comprehensive **Multimodal Fake News Detection System** that checks the authenticity of news articles using both **text** (BERT) and **images** (ResNet). It leverages deep learning techniques to analyze semantic integrity and visual features to classify news as **Real** or **Fake**.

## 🚀 Features

*   **Multimodal Analysis**: Combines text and image features for higher accuracy.
*   **BERT-based Text Model**: Uses a pre-trained `bert-base-uncased` model for state-of-the-art text understanding.
*   **ResNet-based Image Model**: Utilizes `ResNet50` (or similar CNN architecture) to extract visual features.
*   **User-Friendly App**: Interactive Streamlit web application for real-time detection.
*   **Custom Dataset Support**: Easy-to-use CSV format for training on your own datasets.

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
