import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Change this based on your needs
model = SentenceTransformer(MODEL_NAME)

# Load CSV
csv_path = "data/dog_breeds.csv"  # Update with your actual CSV file path
df = pd.read_csv(csv_path)

# Specify the column to encode
text_column = "description"  # Replace with actual column name

# Ensure no NaN values in the text column
df[text_column] = df[text_column].fillna("")

# Generate embeddings
def get_embedding(text):
    return model.encode(text, convert_to_numpy=True).tolist()  # Convert to list for easy storage

df["embeddings"] = df[text_column].apply(get_embedding)

# Save to CSV (JSON format for embeddings) or Parquet (efficient binary format)
# df.to_csv("embedded_data.csv", index=False)
df.to_parquet("embedded_data.parquet", index=False)  # Recommended for large datasets

print("Embeddings saved successfully!")
