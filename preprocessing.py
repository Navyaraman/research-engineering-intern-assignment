import pandas as pd
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Load JSONL dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line)
                data.append(record.get('data', record))  # Handle different JSON structures
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:100]}...")
    return pd.DataFrame(data)

# Clean HTML tags and unnecessary symbols
def clean_text(text):
    if not isinstance(text, str) or text.lower() in ['[deleted]', '[removed]']:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Extract domain from URL
def extract_domain(url):
    if isinstance(url, str):
        return urlparse(url).netloc
    return None

# Preprocess dataset
def preprocess_data(df):
    expected_columns = ['subreddit', 'title', 'selftext', 'upvote_ratio', 'score', 'created_utc', 'url', 'author']
    df = df[[col for col in expected_columns if col in df.columns]]
    
    # Convert timestamp to datetime
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')

    # Fill missing values and clean text
    text_cols = ['title', 'selftext']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)

    # Normalize subreddit names
    if 'subreddit' in df.columns:
        df['subreddit'] = df['subreddit'].str.lower()

    # Convert upvote_ratio and score to numeric
    for col in ['upvote_ratio', 'score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Extract hashtags and keywords
    df['hashtags'] = df.apply(lambda row: re.findall(r'#(\w+)', row['title'] + ' ' + row['selftext']), axis=1)
    df['keywords'] = df.apply(lambda row: re.findall(r'\b\w{4,}\b', (row['title'] + ' ' + row['selftext']).lower()), axis=1)

    # Extract domain from URL
    if 'url' in df.columns:
        df['domain'] = df['url'].apply(extract_domain)

    return df

# Load and preprocess data
file_path = "data.jsonl"  # Update if needed
df = load_jsonl(file_path)
df_cleaned = preprocess_data(df)

# Save cleaned data
df_cleaned.to_csv('cleaned_data.csv', index=False)
df_cleaned.to_json('cleaned_data.jsonl', orient='records', lines=True)

print("✅ Preprocessing complete. Data saved as 'cleaned_data.csv' and 'cleaned_data.jsonl'.")
