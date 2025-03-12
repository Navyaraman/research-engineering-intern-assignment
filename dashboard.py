# Install before running: pip install streamlit pandas matplotlib seaborn wordcloud bertopic networkx

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
import os

# Optional: Load topic labels from BERTopic model (if saved)
def load_topic_labels():
    try:
        from bertopic import BERTopic
        topic_model = BERTopic.load("bertopic_model")
        info = topic_model.get_topic_info()
        return info.set_index("Topic")["Name"].to_dict()
    except:
        return {}

# Load final merged data
@st.cache_data
def load_data():
    df = pd.read_csv("final_dashboard_data.csv", parse_dates=["created_utc"])
    label_map = load_topic_labels()
    if "topic" in df.columns and label_map:
        df["topic_label"] = df["topic"].map(label_map)
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("🔍 Filter Options")
subreddits = df["subreddit"].dropna().unique()
sentiments = df["title_sentiment"].dropna().unique()
topics = df["topic_label"].dropna().unique() if "topic_label" in df.columns else df["topic"].dropna().unique()

selected_subreddits = st.sidebar.multiselect("Subreddit", subreddits)
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments)
selected_topics = st.sidebar.multiselect("Topic", topics)

# Apply filters
filtered_df = df.copy()
if selected_subreddits:
    filtered_df = filtered_df[filtered_df["subreddit"].isin(selected_subreddits)]
if selected_sentiments:
    filtered_df = filtered_df[filtered_df["title_sentiment"].isin(selected_sentiments)]
if selected_topics:
    topic_col = "topic_label" if "topic_label" in filtered_df.columns else "topic"
    filtered_df = filtered_df[filtered_df[topic_col].isin(selected_topics)]

st.title("📊 AI-Powered Social Media Dashboard")
st.write("Explore trends, topics, sentiment, summaries, and network connections from Reddit data using AI & NLP.")
# 🧠 BERTopic revealed these dominant narratives:
bertopic_insights = [
    "🏛️ Government accountability and oversight",
    "📰 Media bias and the rise of fake news",
    "💰 Economic impacts of major policy decisions",
    "✊ Social justice movements and civil liberties"
]


st.markdown("""


### 🔍 What This Dashboard Reveals

**📅 When Did Reddit Erupt?**  
Discover the political flashpoints that triggered spikes in discussion—protests, policy shifts, and controversies that lit up Reddit like wildfire.

**🧵 What’s Everyone Talking About?**  
Unravel the most talked-about threads—think veterans, nukes, media distrust, and democracy—all decoded through cutting-edge topic modeling.

**😊 How Does Reddit *Feel* About It All?**  
Track the emotional heartbeat of the platform—rising optimism, sharp sarcasm, deep frustration, and the occasional meme-driven rage.

**⚠️ Misinformation or Misunderstood?**  
Peek into potential red flags—conspiracy-coded keywords and contentious claims—and see how Reddit communities push back with fact-checking firepower.

**🏆 Which Subreddits Rule the Conversation?**  
Meet the power players. From heated hubs like *r/politics* to the long-comment titans in *r/neoliberal*, see where influence lives and narratives spread.

**🌐 Who Talks to Whom?**  
Zoom into hashtag highways—where left-leaning groups form buzzing intersections and right-leaning spaces build quieter echo chambers.


## 📚 Reddit Political Discourse: An AI-Powered Analysis

This project dives deep into Reddit’s political landscape using NLP, AI, and network analysis to unravel trends, sentiments, and misinformation signals across thousands of posts.

#### 🔥 Surge in Activity Driven by Political Events

A clear spike in user activity correlates with politically charged events—particularly around controversial decisions like nuclear staff layoffs and FAA dismissals. Reddit becomes a real-time barometer of political anxiety, with a flood of posts and reactions centered around these moments. Subreddits like r/politics, r/democrats, and r/worldpolitics emerged as central hubs, consistently driving engagement.

#### 💬 Sentiment Reflects Polarization

Sentiment analysis on titles and post content reveals a divided audience. While some posts exude optimism or support, many titles carry frustration, distrust, or sarcasm—especially toward specific political figures. Posts flagged as negative sentiment were often those involving layoffs, misinformation claims, or partisan decisions, highlighting emotional reactions to governance.

#### 🧠 Topic Modeling Reveals Key Narratives

Using BERTopic, we uncovered dominant narratives including: """)
for point in bertopic_insights:
    st.markdown(f"- {point}")
st.markdown("""

These clusters weren’t isolated—they reflect a vibrant, contentious debate where Reddit acts as a decentralized forum for public discourse.

#### 🧵 Keyword & Hashtag Analysis: What's Being Talked About

Keyword frequency and TF-IDF analysis show a dataset saturated with references to Trump, veterans, and nuclear staff. Words like “people”, “know”, and “want” reflect an audience engaged in active questioning. Interestingly, hashtags were rarely used, suggesting that Redditors rely more on context-rich conversation than viral tagging.

#### 📢 Most Engaging Posts: High-Impact News Drives Attention

The most upvoted and most commented posts center around politically sensitive topics—staff firings, perceived injustice, and discussions about democracy. These posts aren't just popular; they serve as lightning rods for debate, especially in r/neoliberal, which hosts massive discussion threads with thousands of comments daily.

#### ⚠️ Misinformation Signals: Visible, but Contained

While not dominant, potential misinformation does appear—often tied to conspiracy-related keywords like “hoax” or “false claim”. Posts flagged as “Check-worthy” generated polarized sentiment but also drew fact-checks and rebuttals, showcasing Reddit's self-regulating community behavior.

#### 🌐 Hashtag Network Graph: Community Polarization Visualized

The hashtag co-occurrence network highlights a compelling structural insight:

Left-leaning subreddits (like r/liberal, r/worldpolitics, r/anarchism) form tightly connected clusters through shared language and topics.

Right-leaning subreddits (like r/conservative, r/republican) appear more isolated, with limited shared hashtags outside their ideological space.

This suggests strong community polarization—mirroring offline political divides.

##### 🧩 Conclusion: Reddit as a Mirror to Sociopolitical Sentiment

This dashboard doesn't just visualize Reddit data—it tells the story of political discourse in motion. The platform serves as a chaotic yet insightful pulse of public sentiment, shaped by news cycles, user emotion, and evolving narratives.

Through AI summaries, topic modeling, and network graphs, we've surfaced meaningful insights that highlight the power—and complexity—of digital dialogue in modern democracy.

""")
# Tabs layout
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visual Insights", "🧠 AI Summaries", "⚠️ Misinformation", "🌐 Hashtag Network"])

# ------------------ TAB 1 ------------------
with tab1:
    st.subheader("📅 Posts Over Time")
    if not pd.api.types.is_datetime64_any_dtype(filtered_df["created_utc"]):
        filtered_df["created_utc"] = pd.to_datetime(filtered_df["created_utc"], errors='coerce')
    if not filtered_df.empty:
        ts = filtered_df.resample("D", on="created_utc").size()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts.index, ts.values, marker="o")
        ax.set_title("Post Activity Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No data available for the selected filters.")

    st.subheader("😊 Sentiment Distribution")
    if not filtered_df.empty:
        sentiment_counts = filtered_df["title_sentiment"].value_counts()
        if not sentiment_counts.empty:
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["green", "red", "gray"])
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No sentiment data available.")

    st.subheader("🏆 Top 10 Subreddits")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        filtered_df["subreddit"].value_counts().nlargest(10).plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title("Top 10 Active Subreddits")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("🌍 Most Shared Domains")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        filtered_df["domain"].value_counts().nlargest(10).plot(kind="bar", color="orange", ax=ax)
        ax.set_title("Most Shared Domains")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("💬 Word Cloud from Titles")
    if not filtered_df.empty:
        text = " ".join(filtered_df["title"].dropna())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="viridis").generate(text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Not enough text for word cloud.")

# ------------------ TAB 2 ------------------
with tab2:
    st.subheader("🧠 AI-Generated Post Summaries")
    if not filtered_df.empty:
        cols = ["title", "summary", "title_sentiment", "topic_label"] if "topic_label" in filtered_df.columns else ["title", "summary", "title_sentiment", "topic"]
        st.dataframe(filtered_df[cols].head(10))
        st.download_button("📥 Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_data.csv", mime="text/csv")
    else:
        st.info("No summaries to display.")

# ------------------ TAB 3 ------------------
with tab3:
    if "misinformation_flag" in filtered_df.columns:
        st.subheader("⚠️ Posts Marked as Check-worthy")
        flagged = filtered_df[filtered_df["misinformation_flag"] == "Check-worthy"]
        if not flagged.empty:
            st.dataframe(flagged[["title", "misinformation_flag"]].head(5))
        else:
            st.info("No flagged posts in current selection.")

# ------------------ TAB 4 ------------------
with tab4:
    st.subheader("🌐 Hashtag Network Graph Between Subreddits")
    if not filtered_df.empty:
        G = nx.Graph()

        # Extract hashtags from titles
        filtered_df["hashtags"] = filtered_df["title"].str.findall(r"#(\w+)")

        # Build graph: connect subreddits sharing the same hashtags
        for tag in filtered_df.explode("hashtags")["hashtags"].dropna().unique():
            subs = filtered_df[filtered_df["title"].str.contains(f"#{tag}", case=False, na=False)]["subreddit"].unique()
            for i in range(len(subs)):
                for j in range(i+1, len(subs)):
                    G.add_edge(subs[i], subs[j], label=tag)

        if G.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            pos = nx.spring_layout(G, k=0.5)
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000, alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.4)
            nx.draw_networkx_labels(G, pos, font_size=10)
            ax.set_title("Subreddits Sharing Common Hashtags")
            st.pyplot(fig)
        else:
            st.info("No shared hashtags found among selected subreddits.")
    else:
        st.info("No data to visualize the network.")

    # Caption
    st.markdown("""
    ### 📌 Subreddits Sharing Common Hashtags
    **Insight:**  
    This network graph shows how subreddits are interconnected based on common hashtags in post titles.

    - **Nodes** represent subreddits.  
    - **Edges** represent shared hashtags.  
    - The central position of **‘liberal’** suggests it bridges conversations across political lines.  
    - Subreddits like **‘anarchism’**, **‘democrats’**, and **‘worldpolitics’** form a tight cluster, showing more hashtag overlap.  
    - On the other hand, **‘conservative’** appears more isolated, only sharing hashtags with **‘republican’**, indicating fewer shared tags with the broader network.
    """)
