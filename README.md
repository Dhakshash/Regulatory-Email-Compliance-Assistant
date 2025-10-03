# 📧 Regulatory Email Compliance Assistant  

An AI-powered assistant that checks financial emails for **FCA COBS 4 compliance**.  
The system flags risky phrases, cites relevant handbook sections, and rewrites emails to make them compliant — while preserving factual details.  

---

## 🎯 Problem Statement  
Financial institutions must ensure all communications with clients comply with **FCA Handbook COBS 4**.  
Manual reviews are:  
-  Time-consuming  
-  Error-prone  
-  Inconsistent  

This project automates compliance checks, reduces risk, and speeds up communication workflows.  

---

## 🚀 Features  

### 🔹 Core Capabilities  
- **Retrieval-Augmented Generation (RAG)**  
  - FAISS vector index built from **FCA Handbook COBS 4 extracts**.  
  - Embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`.  

- **Retriever + QA Chain**  
  - Retriever: FAISS index queried with **Maximal Marginal Relevance (MMR)** for diverse, relevant context.  
  - QA: `RetrievalQA` chain in **LangChain**, combining handbook context + user email → structured compliance JSON.  

- **LLM-Powered Analysis**  
  - **Meta LLaMA-3.1 8B Instruct** via Hugging Face pipelines.  
  - Enforced **JSON-only outputs** with decision, cited sections, and compliant rewrite.  

- **Compliance Logic**  
  - Flags prohibited guarantees (`"guaranteed"`, `"no risk"`, `"risk-free"`).  
  - Inserts disclaimers only if **verbatim in context**.  
  - Keeps factual details unchanged (dates, %, names).  

---

### 🔹 Interfaces  
- **Streamlit App** → paste any email draft and run compliance check.  
- **Gmail Integration** → fetch Inbox/Drafts/Sent, check compliance, save rewrites back to Gmail.  

---

## 📂 Repository Contents  
- `Regulatory_Email_Compliance_Assistant_.ipynb` → Colab notebook (RAG pipeline with Retriever + QA).  
- `app_email_compliance_check_.py` → Streamlit app UI.  
- `gmail_compliance_assistance.py` → Gmail integration script.  
- `Compliance Assistant.pdf` → Demo screenshot of Streamlit app.  

---

## Architecture

flowchart TD

     A[Email Input (Streamlit/Gmail)] --> B[Embeddings (MiniLM)]
     B --> C[FAISS Vector Index]
     C --> D[Retriever (MMR Search)]
     D --> E[LangChain RetrievalQA Chain]
     E --> F[LLaMA 3.1 8B Instruct]
     F --> G[JSON Compliance Output]
     G --> H[Streamlit UI / Gmail Draft]

---

## 📌 Notes
Requires Hugging Face access token for LLaMA 3.1.

FAISS index must be prebuilt from FCA Handbook (COBS 4).

Best run on GPU (bitsandbytes supported).

Tested on Python 3.10/3.11.

---

## Future Enhancements
✅ Deploy Streamlit app to cloud.

✅ Extend to more FCA Handbook sections.

🚀 Batch email compliance checks.

🚀 Compliance confidence scoring.

🚀 Analytics dashboard.
