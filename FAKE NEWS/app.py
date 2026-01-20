import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from src.model import FakeNewsModel
import os

# Page Config
st.set_page_config(
    page_title="Fake News Detector", 
    page_icon="🕵️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Harmonized Glassmorphism (Cyber-Future Theme)
st.markdown("""
<style>
    /* --- Master Palette --- 
       Background: Deep Space (#0f0c29, #302b63, #24243e)
       Accent: Neon Cyan (#00f2fe) to Electric Blue (#4facfe)
       Glass: Transparent White/Blue Tint
       Text: White & Light Gray
    */

    /* Main Background - Animated Deep Space Gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #1a103c, #24243e, #090919);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass Container Style - Unified Tint */
    .glass-container {
        background: rgba(30, 30, 50, 0.3); /* Slightly blue-tinted glass */
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(79, 172, 254, 0.2); /* Border matches accent */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Customizing Streamlit Elements (Sidebar, Toolbar) */
    div[data-testid="stToolbar"], div[data-testid="stDecoration"], div[data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.6) !important; /* Matches background hue */
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(79, 172, 254, 0.1);
    }
    
    /* Input Fields (Text Area) - Harmonized */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.4) !important;
        color: #e0e0ff !important; /* Slight blue tint text */
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(5px);
    }
    .stTextArea textarea:focus {
        border-color: #4facfe !important;
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.4);
    }
    
    /* File Uploader (Drag Box) - Harmonized */
    div[data-testid="stFileUploadDropzone"] {
        background: rgba(30, 40, 70, 0.3) !important;
        backdrop-filter: blur(10px);
        border: 1px dashed rgba(79, 172, 254, 0.5) !important; /* Cyan border */
        border-radius: 15px;
        color: #d0d0ff;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        background: rgba(79, 172, 254, 0.1) !important;
        transform: scale(1.02);
        border-color: #00f2fe !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
    }
    
    /* Buttons - Gradient matching the theme */
    .stButton>button {
        background: linear-gradient(90deg, #1e2029, #2a2d3a);
        color: #4facfe;
        border: 1px solid #4facfe;
        border-radius: 30px;
        padding: 10px 30px;
        font-weight: bold;
        backdrop-filter: blur(5px);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
        border-color: white;
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
        transform: translateY(-2px);
    }

    /* Typography */
    h1, h2, h3 {
        background: -webkit-linear-gradient(#eee, #4facfe); /* Gradient text */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    p, label, li {
        color: #b0b3d6 !important; /* Cool grey text */
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #00f2fe !important; /* Cyan numbers */
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.6);
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0c0 !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>🕵️ GlassVerify AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; margin-bottom: 40px;'>Next-Gen Multimodal Misinformation Detection</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🔮 About")
st.sidebar.info("""
**GlassVerify AI** uses deep learning to detect fake news.
- **BERT** for Text
- **ResNet** for Images
""")
st.sidebar.markdown("---")

# Model Loading
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    model = FakeNewsModel(num_classes=2)
    model_path = 'fake_news_model.pth'
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)
            return model, device
        except Exception as e:
            # st.error(f"Error loading model: {e}")
            return None, device
    else:
        return None, device

model, device = load_model()

# --- Main UI Rendering (Rendered regardless of model status so UI can be seen) ---

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 Analysis Content", unsafe_allow_html=True)
    news_text = st.text_area("Paste Article Text", height=250, placeholder="Type or paste the news content here...", label_visibility="collapsed")

with col2:
    st.markdown("### 🖼️ Visual Evidence", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Evidence', use_column_width=True)
    else:
        st.info("Upload an image to verify.")

st.markdown("---")

# Center Action
_, mid, _ = st.columns([1, 1, 1])
with mid:
    analyze_btn = st.button("✨ Verify Authenticity", use_container_width=True)

# Logic Handling
if analyze_btn:
    if model is None:
        st.warning("⚠️ **Model Weights Not Found!**")
        st.error("The model file `fake_news_model.pth` is missing (deleted to save space).")
        st.info("Please run `python -m src.train` in your terminal to regenerate variables and enable analysis.")
    elif not news_text:
        st.error("Please provide news text to analyze.")
    elif not uploaded_file:
        st.error("Please upload an image to analyze.")
    else:
        with st.spinner("🔮 Consultng the Oracle (running inference)..."):
            try:
                # Preprocess
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Text
                encoding = tokenizer.encode_plus(
                    news_text, add_special_tokens=True, max_length=128,
                    padding='max_length', truncation=True, return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Image
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, img_tensor)
                    probs = F.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probs, 1)
                    
                    label_map = {0: "FAKE", 1: "REAL"}
                    prediction = label_map[predicted_class.item()]
                    
                # Custom Result UI
                st.markdown("<br>", unsafe_allow_html=True)
                if prediction == "REAL":
                    st.success(f"✅ **VERIFIED REAL** ({confidence.item()*100:.1f}% confidence)")
                    st.balloons()
                else:
                    st.error(f"🚨 **POTENTIAL FAKE** ({confidence.item()*100:.1f}% confidence)")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
