# Malayalam Fake News Detection System 📰🇮🇳

This project is a Transformer-based Fake News Detection System for the **Malayalam language**, designed to identify whether a news article is real or fake. It was built as part of an academic project using Hugging Face Transformers, PyTorch, and NLP tools.

---

## 🚀 Features

- ✅ Detects fake news in Malayalam text
- ✅ Fine-tuned Transformer model using Hugging Face
- ✅ Streamlit-based web interface for easy testing
- ✅ Includes training, preprocessing, and deployment code
- ✅ Malayalam dataset included (CSV format)

---

## 🧾 Folder Structure

```
malayalam-fake-news-detector/
├── model/
│   └── checkpoint-2634/
│       ├── config.json
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── app.py
├── app_streamlit.py
├── train.py
├── prepare_dataset.py
├── dataset.csv
├── ma_fake.csv
├── ma_true.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/malayalam-fake-news-detector.git
cd malayalam-fake-news-detector
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

(Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

---

## 🏋️‍♂️ Train the Model

```bash
python train.py
```

This will load and preprocess the dataset (`ma_fake.csv`, `ma_true.csv`) and save the model checkpoint in the `model/` folder.

---

## 🌐 Run the Web App

```bash
streamlit run app_streamlit.py
```

This starts a web interface where you can enter Malayalam text and get real/fake prediction.

---

## 🧠 Model Details

- Model: Fine-tuned multilingual Transformer (BERT-based)
- Library: Hugging Face Transformers
- Tokenizer: SentencePiece / WordPiece (via HF)
- Training Framework: PyTorch

---

## 📊 Dataset

Custom dataset of real and fake Malayalam news articles in CSV format:
- `ma_true.csv`: Real news
- `ma_fake.csv`: Fake news
- `dataset.csv`: Combined version (for training)

---

## 👤 Author

**Jayadev J**
9446059322 
jjayadev2003@gmail.com
B.Tech Final Year Student, Computer Science and Technology  
RHCSA Certified | Python | NLP | Deep Learning

---

## 📜 License

This project is open-source and available for academic/research use.
