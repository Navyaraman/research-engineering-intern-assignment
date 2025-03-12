# Install requirements before running:
# pip install pandas nltk bertopic tqdm transformers sentence-transformers

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import logging

# ------------------ 1. Load Data ------------------
df = pd.read_csv("cleaned_data.csv")

# ------------------ 2. Sentiment Analysis ------------------
print("🔍 Running Sentiment Analysis...")
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return "neutral"
    score = sia.polarity_scores(text)["compound"]
    return "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

df["title_sentiment"] = df["title"].astype(str).apply(get_sentiment)
df["selftext_sentiment"] = df["selftext"].astype(str).apply(get_sentiment)
print("✅ Sentiment Analysis Complete")

# ------------------ 3. Topic Modeling with BERTopic ------------------
print("🧠 Running Topic Modeling...")
df["combined_text"] = df["title"].fillna('') + " " + df["selftext"].fillna('')
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(df["combined_text"])
df["topic"] = topics
topic_model.save("bertopic_model")
print("✅ Topic Modeling Complete")

# ------------------ 4. Misinformation Detection ------------------
print("⚠️ Checking for Misinformation...")
misinfo_keywords = ["fake news", "hoax", "conspiracy", "debunked", "false claim", "misleading"]

def flag_misinformation(text):
    if pd.isna(text) or text.strip() == "":
        return "Not Flagged"
    text = text.lower()
    return "Check-worthy" if any(word in text for word in misinfo_keywords) else "Not Flagged"

df["misinformation_flag"] = df["title"].astype(str).apply(flag_misinformation)
print("✅ Misinformation Detection Complete")

# ------------------ 5. Summarization using T5 ------------------
print("📄 Generating Summaries...")
logging.getLogger("transformers").setLevel(logging.ERROR)
model_name = "t5-small"
summarizer = pipeline("summarization", model=model_name, device=-1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_input_tokens = 512

def generate_summary(text):
    if pd.isna(text) or len(text.strip()) < 30:
        return text
    tokens = tokenizer.encode(text, truncation=True, max_length=max_input_tokens)
    clean_text = tokenizer.decode(tokens, skip_special_tokens=True)
    input_len = len(tokens)
    if input_len < 10:
        return text
    max_len = min(50, max(10, int(input_len * 0.6)))
    min_len = min(10, max_len // 2)
    try:
        summary = summarizer(clean_text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"⚠️ Skipping due to error: {e}")
        return text

tqdm.pandas(desc="Summarizing")
df["summary"] = df["selftext"].astype(str).progress_apply(generate_summary)
print("✅ Summarization Complete")


# ------------------ 6. Save Final Output ------------------
df.to_csv("final_dashboard_data.csv", index=False)
print("✅ All enrichment steps complete! Final file saved as 'final_dashboard_data.csv'")
