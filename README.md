# RAG-Vector-Chunking-Analysis
An end-to-end RAG pipeline evaluating chunking strategies and mitigating LLM hallucinations using LangChain, FAISS, and local open-source models.
# 📚 RAG Pipeline Optimization & Evaluation: Chunking Strategies with Local LLMs

## Overview
This repository demonstrates the end-to-end construction, evaluation, and optimization of a Retrieval-Augmented Generation (RAG) system entirely using open-source, local Large Language Models (LLMs). The project focuses not just on building a functional RAG pipeline, but on rigorously evaluating its retrieval performance and answer quality, specifically tackling the trade-offs between text chunking strategies and LLM hallucinations.

## Key Features
* **Local RAG Pipeline:** Built utilizing LangChain, Hugging Face models, and FAISS for efficient similarity search without relying on paid APIs.
* **Quantitative Evaluation:** Implements a systematic evaluation framework to measure the system's performance across three key metrics:
    * **Retrieval Hit Rate:** Assessing the vector database's ability to fetch relevant context.
    * **Faithfulness:** Evaluating whether the LLM's answers are strictly grounded in the retrieved context to prevent hallucinations.
    * **Answer Relevance:** Ensuring the generated responses directly address the user's queries.
* **Chunking Strategy Experimentation:** A deep dive into how different document chunk sizes (256, 512, 1024) impact retrieval accuracy and response quality.

## Tech Stack
* **Frameworks:** LangChain, Hugging Face Transformers
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `sentence-transformers/all-mpnet-base-v2`
* **Generative Model:** `Qwen/Qwen2.5-0.5B`
* **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

## Key Findings & Insights
Through rigorous testing on domain-specific academic essay data, the project yielded several critical insights for deploying RAG systems:

1.  **Chunk Size vs. Retrieval Precision:** There is a clear negative correlation between chunk size and retrieval hit rate. Smaller chunks (**256 tokens**) provided the highest precision for factual, definition-based queries by creating highly specific mathematical vectors.
2.  **Mitigating Hallucinations:** When using larger chunk sizes (**512 or 1024 tokens**), the vector representations became noisy. This frequently caused the retriever to miss specific information, which in turn forced the LLM to rely on its internal knowledge, leading to severe hallucinations and fabricated details. 
3.  **Optimal Configuration:** For factual QA tasks on academic texts, a smaller chunk size (**256 tokens** with an overlap of **50**) proved to be the safest and most accurate strategy, achieving an 80% retrieval hit rate and significantly reducing ungrounded answers.

## Repository Structure
```text
├── offline_doc/                # Directory containing the source PDF documents
├── RAG_Evaluation_Report.pdf   # Detailed analysis report on metrics and chunking experiments
├── RAG_demo.ipynb              # Main Jupyter Notebook containing the pipeline and evaluation code
├── requirements.txt            # Python dependencies
└── README.md
