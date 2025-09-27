import streamlit as st
import pandas as pd

st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")

st.title("📰 Real-Time News Sentiment Dashboard")

# Load predictions
df = pd.read_csv("predictions.csv")

# Map predictions to labels
df["Sentiment"] = df["predicted_label"].map({0.0: "Negative", 1.0: "Positive"})

# Show summary
st.metric("Total Headlines", len(df))
st.metric("Positive", (df["Sentiment"] == "Positive").sum())
st.metric("Negative", (df["Sentiment"] == "Negative").sum())

# Show table
st.subheader("Latest Headlines")
st.dataframe(df[["text", "Sentiment"]], use_container_width=True)

# Optional: Add chart
st.subheader("Sentiment Distribution")
st.bar_chart(df["Sentiment"].value_counts())

