# Import required libraries
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize the Flask web application
app = Flask(__name__)

# -----------------------------
# Load Pretrained Model & Tokenizer
# -----------------------------
# The model and tokenizer are loaded from the "model" directory
# This should contain config.json, pytorch_model.bin, tokenizer files, etc.
model_path = "model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode (important for inference)

# -----------------------------
# Home Route - Simple HTML Form
# -----------------------------
@app.route('/')
def home():
    # This route displays a simple form to accept Malayalam news input
    return '''
    <h1>BERT Malayalam Fake News Detector</h1>
    <form method="POST" action="/predict">
        <textarea name="text" rows="6" cols="60" placeholder="Enter Malayalam news text..."></textarea><br><br>
        <button type="submit">Predict</button>
    </form>
    '''

# -----------------------------
# Prediction Route - Handles Form Submission
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the form
    text = request.form['text']

    # Tokenize the input using the loaded tokenizer
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Use the model to predict the label (0: Fake, 1: Real)
    with torch.no_grad():  # Disable gradient calculation for faster prediction
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()  # Get the predicted label (0 or 1)

    # Show a user-friendly result based on the prediction
    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    
    # Return result and a back link
    return f"<h2>{result}</h2><a href='/'>Go back</a>"

# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == '__main__':
    # Start the app in debug mode on localhost (http://127.0.0.1:5000)
    app.run(debug=True)
