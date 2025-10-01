📦 Aksum Procurement RAG Chatbot

Retrieval-Augmented Generation (RAG) chatbot for procurement teams at Aksum Trademart Pvt. Ltd.
Built with FAISS + Sentence Transformers + Streamlit, and pluggable LLM support (OpenAI / Hugging Face / Ollama).

Q. What problem does this solve?

A.Cuts manual lookup time in procurement (penalties, GST, approvals), enabling instant answers.

NOTE - This project is uploaded as a prototype to show the main idea of what my company’s tech team and I built, and to highlight the parts I personally worked on. I can’t share the full repository because of confidentiality and agreements with my company.

🚀 Project Overview

Procurement teams often waste hours searching through:

Supplier contracts (delivery penalties, payment terms)

Procurement policies (credit norms, GST rules, return windows)

Invoices & POs (payment cycles, HSN codes)

Internal FAQs

This chatbot answers procurement queries in seconds by combining:

Document retrieval (via FAISS + embeddings)

Pin-point extractive answers (regex + semantic search)

Optional LLM polish (OpenAI, Hugging Face, or Ollama for natural answers + citations)

👉 Goal: Reduce query resolution time, increase transparency, and empower procurement staff to self-serve.

✨ Features

📂 Document ingestion & chunking → PDF, text, CSV, FAQs

🔍 Semantic search (FAISS + MiniLM) → retrieve most relevant policy/contract chunks

📝 Pin-point extractive answers (regex rules for penalties, GST, credit terms)

🤖 Optional LLM polish (configurable: OpenAI, Hugging Face, or Ollama)

📚 Citations → every answer links to source docs

🖥️ Streamlit UI → upload docs, rebuild index, ask questions

📊 Logging → every query/answer logged for monitoring

🏗️ Architecture
flowchart TD
  A[User Query] --> B[Embeddings Search (FAISS)]
  B --> C[Top-k Document Chunks]
  C --> D[Extractive Pin-point Answer Builder]
  D --> E[LLM (optional polish)]
  E --> F[Final Answer + Citations]
  F --> G[Streamlit UI Display]


Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)

Index: FAISS FlatIP (cosine similarity)

Extractive fallback: regex + overlap heuristics

LLM layer: Plug-in → OpenAI GPT, Hugging Face pipeline, or Ollama (local Mistral/LLM3/phi3-mini)

Example Query:

“What is the penalty for late delivery of TMT bars?”

Answer:

Late delivery penalty is 0.5% of PO value per day, capped at 5% [1][2].

Sources:

Supplier Agreement – Shakti Steels Pvt. Ltd.

Procurement Policy v1.1

⚙️ Tech Stack

Language: Python 3.11

Frameworks: Streamlit, FAISS, Hugging Face Transformers

LLM (optional):

OpenAI GPT (if API key available)

Hugging Face local pipeline (Mistral / Zephyr)

Ollama (local mistral, llama3, phi3-mini)

Storage: Local FAISS index + JSON metadata

🔮 Future Work

Vector DB integration (Pinecone/Weaviate)

Role-based access control

PDF & image OCR ingestion

Analytics dashboard for query trends
