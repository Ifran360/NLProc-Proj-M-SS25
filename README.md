# RAG Project – Summer Semester 2025

## Overview

This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure

- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results
- `utils/`: Helper functions shared across code

## Getting Started

1. Clone the repo
2. `cd baseline/`
3. Install dependencies: `pip install -r ../requirements.txt`

## How Vector Search Works

Traditional keyword-based search systems match exact terms between a query and a document. In contrast, vector search uses machine learning models to capture the **semantic meaning** of text.

In this project, we use a pre-trained `SentenceTransformer` to convert each document and query into a **dense vector** (embedding) in a high-dimensional space. These embeddings are then stored in a **FAISS index**, which enables fast similarity search.

At retrieval time, the user’s query is also embedded, and FAISS searches for vectors in the index that are **closest** (by Euclidean distance or cosine similarity). This allows us to retrieve relevant documents even when the **query and document use different wording**, as long as their meanings are similar.

This approach forms the core of **Retrieval-Augmented Generation (RAG)**, where high-quality retrieved documents can be used to generate accurate and context-aware answers.

## Teams & Tracks
