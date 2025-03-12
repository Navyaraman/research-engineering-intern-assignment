# 🤖 AI Reddit Analyzer  
> **Powered by LLMs, BERTopic, and NLP — for Reddit trend exploration**

---

## 🚀 Overview

**AI Reddit Analyzer** is an end-to-end NLP-powered dashboard built as part of the **SimPPL Research Engineering Intern assignment**.  
It explores Reddit political discourse using **sentiment analysis, topic modeling, TextRank summarization, misinformation tagging**, and visual storytelling — all inside a clean **Streamlit** interface.

---

## 🔗 Live App

🌐 [Launch the Streamlit Dashboard](https://navyaraman-research-engineering-intern-assignment.streamlit.app)

---

## 🗂️ Project Structure

| File | Purpose |
|------|---------|
| `preprocessing.py`             | Cleans the raw `data.jsonl` into `cleaned_data.csv` |
| `prepare_dashboard_data.py`    | Applies AI/ML techniques and outputs `final_dashboard_data.csv` |
| `dashboard.py`                 | Interactive Streamlit dashboard with filters and visualizations |
| `final_dashboard_data.csv`     | Final processed dataset used in the dashboard |
| `requirements.txt`             | Python dependencies |
| `README.md`                    | You’re reading it ✅ |

---

## 🎯 Features

### 📊 Visual Insights
- 📅 **Time-Series Plot** – Reddit post volume over time
- 😊 **Sentiment Pie Chart** – VADER-based sentiment
- 🏆 **Top Subreddits** – Most active political communities
- 🌍 **Most Shared Domains** – Source credibility tracking
- 💬 **Word Cloud** – Dominant discussion terms

### 🧠 AI & NLP Modules
- **Sentiment Analysis** using VADER (NLTK)
- **Topic Modeling** with BERTopic
- **Summarization** using TextRank (`sumy`)
- **Misinformation Detection** with keyword tagging
- **Hashtag Network Graph** to visualize community overlap

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard Main](research-engineering-intern-assignment/IMAGE1.png)

### Hashtag Network Graph
![Network Graph](research-engineering-intern-assignment/IMAGE2.png)


---

## 🧠 Thought Process & Design

- 🧪 VADER was used due to its suitability for social media sentiment
- 🔍 BERTopic captures nuanced themes using embeddings + c-TFIDF
- 🧵 TextRank (via `sumy`) enables AI summarization without needing OpenAI APIs
- 🧠 NetworkX builds hashtag co-occurrence networks to surface subreddit interconnectivity

The system is built for **scalability, interpretability, and insight generation**.

---

## 📂 Dataset

- 📥 Reddit posts in `.jsonl` format  
- 🔗 [Download Sample Data](https://drive.google.com/file/d/1XHtTnUpTjUIIREKGF8ETaZ_hEtxtPJWY/view?usp=drive_link)
- Cleaned to: `cleaned_data.csv`  
- Final enriched version: `final_dashboard_data.csv`

---

## 🛠️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Navyaraman/research-engineering-intern-assignment
cd research-engineering-intern-assignment

# Install dependencies
pip install -r requirements.txt

# Step 1: Clean raw data (optional if already cleaned)
python preprocessing.py

# Step 2: Enrich with AI insights
python prepare_dashboard_data.py

# Step 3: Launch dashboard
streamlit run dashboard.py
