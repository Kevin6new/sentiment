import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
data = pd.read_pickle("preprocessed_data.pkl")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = data['EmbedText'].tolist()
splits = text_splitter.create_documents(texts)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_prompt(tweet_text, retrieved_docs):
    context = "\n\n".join(f"Similar tweet: {doc}" for doc in retrieved_docs)
    prompt_text = (
        f"Given the context of similar tweets below, analyze the sentiment of the following tweet and provide ONLY the sentiment (positive, negative, or neutral) and a star rating from 0.0 to 5.0 (with one decimal place) based on the sentiment:\n\n"
        f"{context}\n\n"
        f"Tweet to analyze: '{tweet_text}'\n\n"
        f"Response:"
    )
    return prompt_text

def classify_sentiment(tweet_text):
    retrieved_docs = retriever.invoke(tweet_text, top_k=5)
    formatted_docs = format_docs(retrieved_docs)
    custom_prompt = generate_prompt(tweet_text, formatted_docs)
    result = llm.invoke(custom_prompt)
    return result.content

st.title("Sentiment Analysis")

tweet_text = st.text_input("Enter a tweet for sentiment analysis:")
if st.button("Analyze Sentiment"):
    if tweet_text:
        sentiment_prediction = classify_sentiment(tweet_text)
        st.write("Sentiment and Star Rating:")
        st.write(sentiment_prediction)
    else:
        st.error("Please enter a valid tweet.")
