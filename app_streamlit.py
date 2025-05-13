# ----------------------------------------
# Import necessary libraries
# ----------------------------------------
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ----------------------------------------
# Set the path to the saved model and tokenizer
# This directory should contain config.json, pytorch_model.bin, vocab.txt, etc.
# ----------------------------------------
model_path = "./model"  # Directory where the model and tokenizer are saved

# ----------------------------------------
# Load the tokenizer and the fine-tuned BERT model
# ----------------------------------------
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# ----------------------------------------
# Configure the Streamlit app's appearance
# ----------------------------------------
st.set_page_config(page_title="Malayalam Fake News Detector", layout="centered")

# ----------------------------------------
# UI Elements - Title and Instructions
# ----------------------------------------
st.title("📰 Malayalam Fake News Detector")
st.markdown("Enter Malayalam news text below to check if it's Fake or Real.")

# ----------------------------------------
# Text area for user input
# ----------------------------------------
user_input = st.text_area("✍️ Enter Malayalam news here:")

# ----------------------------------------
# When the "Check" button is pressed
# ----------------------------------------
if st.button("Check"):
    if user_input.strip() == "":
        # Show a warning if the input is empty
        st.warning("Please enter some text.")
    else:
        # Set the model to evaluation mode (no training)
        model.eval()

        # Tokenize the input news text
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512  # BERT supports up to 512 tokens
        )

        # Disable gradient calculation for faster inference
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()  # 0 = Fake, 1 = Real

        # Show the result
        if prediction == 1:
            st.success("✅ This looks like Real News.")
        else:
            st.error("❌ This might be Fake News.")
