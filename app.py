import streamlit as st
import pandas as pd
import requests
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# -------------------------------
# Spark session
# -------------------------------
spark = SparkSession.builder.appName("NewsSentimentML").getOrCreate()

# -------------------------------
# Training data (for ML pipeline)
# -------------------------------
train_data = [
    ("I love the new product launch", "Positive"),
    ("The stock market crashed today", "Negative"),
    ("The movie was fantastic", "Positive"),
    ("I am very disappointed by the service", "Negative"),
    ("Elections bring uncertainty to the market", "Negative"),
    ("This sports event is amazing", "Positive")
]

df_train = spark.createDataFrame(train_data, ["text", "label"])

# -------------------------------
# ML Pipeline
# -------------------------------
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
lr = LogisticRegression(featuresCol="features", labelCol="labelIndex")

pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, label_indexer, lr])
model = pipeline.fit(df_train)

# Converter from numeric prediction to label
label_converter = IndexToString(inputCol="prediction", outputCol="predicted_label",
                                labels=model.stages[3].labels)

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard (PySpark ML)")

API_KEY = os.environ.get("NEWS_API_KEY")
URL = f'https://newsdata.io/api/1/news?apikey={API_KEY}&language=en'

# Placeholder for live updates
placeholder = st.empty()

# -------------------------------
# Real-time news loop
# -------------------------------
while True:
    try:
        response = requests.get(URL)
        data = response.json()
        articles = data.get('results', [])
        texts = [(article['title'] + " " + (article.get('description') or "")) for article in articles]

        if texts:
            df_news = spark.createDataFrame([(text,) for text in texts], ["text"])
            predictions = model.transform(df_news)
            predictions = label_converter.transform(predictions)
            pdf = predictions.select("text", "predicted_label").toPandas()

            # Display in Streamlit
            with placeholder.container():
                st.subheader("Latest News Headlines with Sentiment")
                st.dataframe(pdf)

                st.subheader("Sentiment Counts")
                st.bar_chart(pdf['predicted_label'].value_counts())

        # Refresh every 30 seconds
        time.sleep(30)

    except Exception as e:
        st.error(f"Error fetching news: {e}")
        time.sleep(30)

