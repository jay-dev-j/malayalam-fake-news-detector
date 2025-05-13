import pandas as pd  # Import the pandas library for data manipulation

# Load both datasets (fake and true news) from CSV files
df_fake = pd.read_csv("ma_fake.csv")  # Load the fake news dataset
df_true = pd.read_csv("ma_true.csv")  # Load the real news dataset

# Add a label column: 1 for fake news, 0 for real news
df_fake['label'] = 1  # Assign label '1' for fake news
df_true['label'] = 0  # Assign label '0' for real news

# Combine the two datasets (fake and true) into one dataset
df = pd.concat([df_fake, df_true], ignore_index=True)  # Concatenate the datasets, resetting the index

# Ensure the column with the text is named 'text'
# Replace 'your_text_column' with the actual column name in your CSV if needed
df = df.rename(columns={"your_text_column": "text"})  # Rename the text column to 'text' for uniformity

# Drop any rows with missing values (to ensure clean data)
df = df.dropna()  # Remove rows that have NaN or missing values

# Save the final dataset to a new CSV file
df.to_csv("dataset.csv", index=False)  # Save the combined and cleaned dataset to 'dataset.csv' without row index

print("✅ dataset.csv created successfully!")  # Print a success message once the process is complete
