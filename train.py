# ------------------------------------------
# Import necessary libraries
# ------------------------------------------
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

# ------------------------------------------
# Load dataset from CSV file
# The CSV file must have two columns: 'text' and 'label'
# 'text' contains the news content, 'label' is 0 for fake, 1 for real
# ------------------------------------------
df = pd.read_csv("dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# ------------------------------------------
# Load the tokenizer for BERT multilingual model
# This will convert text into token IDs that the BERT model can understand
# ------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# ------------------------------------------
# Tokenize the text data
# - `truncation=True`: Truncate texts longer than 512 tokens
# - `padding=True`: Pad shorter texts so they are the same length
# ------------------------------------------
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Convert tokenized data to PyTorch tensors
input_ids = torch.tensor(encodings["input_ids"])
attention_mask = torch.tensor(encodings["attention_mask"])
labels = torch.tensor(labels)

# ------------------------------------------
# Create a custom PyTorch Dataset class
# This allows us to pass tokenized inputs and labels into the model during training
# ------------------------------------------
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# ------------------------------------------
# Split data into training and validation sets
# 90% for training, 10% for validation
# ------------------------------------------
train_ids, val_ids = train_test_split(range(len(labels)), test_size=0.1)
train_dataset = NewsDataset(input_ids[train_ids], attention_mask[train_ids], labels[train_ids])
val_dataset = NewsDataset(input_ids[val_ids], attention_mask[val_ids], labels[val_ids])

# ------------------------------------------
# Load the pre-trained BERT model for classification
# Set `num_labels=2` since we have two classes: fake (0) and real (1)
# ------------------------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# ------------------------------------------
# Define training parameters
# - `output_dir`: directory to save model checkpoints
# - `num_train_epochs`: number of times to go through the full training dataset
# - `per_device_train_batch_size`: number of samples per training step
# - `evaluation_strategy`: evaluate model every epoch
# - `save_strategy`: save model every epoch
# - `logging_dir`: directory to save logs
# ------------------------------------------
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

# ------------------------------------------
# Initialize the HuggingFace Trainer
# It takes care of training, evaluation, logging, and saving
# ------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ------------------------------------------
# Start training the model
# ------------------------------------------
trainer.train()

# ------------------------------------------
# Save the final model and tokenizer to disk
# These will be used later for inference (prediction)
# ------------------------------------------
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
