# E-commerce Multi-Source Product Review Analysis & QA Bot

## 🧾 Overview

This project tackles the challenge of understanding customer feedback from diverse sources — including **Amazon**, **YouTube**, and **Reddit** — by aggregating, clustering, and analyzing product reviews across platforms. 

In addition to review aggregation and topic modeling, this system also implements a **QA bot** that allows users to ask product-specific questions and receive context-driven answers sourced from actual user reviews using a Retrieval-Augmented Generation (RAG) architecture.

The result is an end-to-end product insight system designed to help:
- Consumers make better purchase decisions
- Product managers detect sentiment patterns
- Analysts discover trending product themes

## 🚧 Problem Statement

Product reviews are scattered across platforms and filled with noise, redundancy, and unstructured sentiment. While individual reviews are helpful, they lack **aggregated insight** or structured Q&A capabilities.

Challenges addressed:
- 🚫 Inconsistent review formats across platforms (text, video, Reddit posts)
- 🧩 Difficulty finding *thematic consensus* across thousands of reviews
- ❓ Inability to ask contextual questions like:  
   *“What do users say about battery life?”* or *“Is the camera good in low light?”*

This project solves those gaps using:
- Multi-source scraping
- Topic modeling and clustering
- QA via semantic search + LLM-generated answers

## 🧠 Project Pipeline

### 🛒 Data Collection:
- **YouTube transcripts** scraped via `youtube_transcript_api`
- **Amazon reviews** (pre-collected dataset)
- **Reddit product threads** using `PRAW`

### 🔍 Review Preprocessing:
- Regex cleaning and normalization
- Sentence filtering
- Deduplication and source labeling

### 🧾 QA Bot (RAG):
- FAISS vector store for document retrieval
- LangChain/OpenAI for contextual QA generation
- Top-k similarity-based passage ranking

> Result: Ask “What do people dislike about this product?” and get a grounded, review-based answer.

## 🧪 Example QA Interactions

**Query:** _“What do users say about battery life of iPhone 15?”_  
**Answer:**  
- Most users report that the battery lasts about 6–8 hours of heavy use.  
- Several Reddit posts highlight fast battery drain after 1 year.  
- YouTube reviewers suggest turning off background sync to save power.  

---

**Query:** _“Is the camera good in low light?”_  
**Answer:**  
- Amazon reviews frequently mention poor detail in night shots.  
- Reddit users recommend using manual mode for better results.  
- A YouTube reviewer compares it unfavorably with a mid-tier DSLR.

## 🎯 Key Features

- 🔄 **Multi-Source Scraping**: Review data from Reddit, YouTube, and Amazon
- 🧼 **Smart Preprocessing**: Text cleaning, deduplication, and sentence-level segmentation
- 🧠 **Topic Modeling with BERTopic**: Identify pain points and praise patterns
- 🤖 **QA Bot Powered by RAG**: Ask questions and get real, evidence-based answers
- ⚡ **Modular Design**: Each step can be used independently or chained
