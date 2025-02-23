# Intelligent Dog Breed Assistant
![Project Screenshot](images/screenshot.png)

## Overview

This project is a **Dog Breed Assistant** that allows users to ask questions about dog breeds and retrieve relevant information using **Elasticsearch** and **OpenAI's GPT models**. The system employs hybrid search techniques, including **BM25 keyword search, vector search, and semantic reranking**.

## Features

- **FastAPI Backend**: Handles API requests and integrates with Elasticsearch.
- **Elasticsearch Indexing & Retrieval**: Enables efficient keyword and vector-based searches.
- **LangChain-based Agent**: Uses a retrieval-augmented generation (RAG) approach for answering queries.
- **Streamlit Frontend**: Provides a simple UI for user interaction.
- **Precomputed Embeddings**: Improves search efficiency by embedding dog breed descriptions beforehand.
- **Dockerized Deployment**: Ensures easy setup and portability using `docker-compose`.

---

## Technology Stack

### **1. FastAPI**

- Provides a lightweight API for handling user queries.
- Manages Elasticsearch interactions and query processing.

### **2. Elasticsearch**

- Stores dog breed information and enables powerful search capabilities.
- Supports **BM25 keyword search**, **vector-based similarity search**, and **hybrid search with reranking**.
- Uses a **Transformer model (MiniLM-L6-v2)** for text similarity.

### **3. LangChain**

- Used for building an intelligent agent that can:
  - Retrieve information from Elasticsearch.
  - Perform **question-answering** using OpenAI models.
  - Combine multiple tools (e.g., data analysis, search retrieval).

### **4. SentenceTransformers**

- Precomputes embeddings for dog breed descriptions to enable vector search.
- Uses **`all-MiniLM-L6-v2`**, a compact and efficient embedding model.

### **5. Streamlit**

- Provides a web UI for users to ask questions.
- Calls the FastAPI backend to fetch answers.
- Displays response history for better user experience.

### **6. Docker & Docker Compose**

- Simplifies deployment by containerizing the backend, frontend, and Elasticsearch.
- Ensures reproducibility across different environments.

---

## Architecture & Workflow

### **1. Data Preparation & Embeddings**

- The `precompute_embeddings.py` script reads dog breed descriptions and generates **vector embeddings** using SentenceTransformers.
- The embeddings are saved in `embedded_data.parquet` for faster loading.

### **2. FastAPI Backend (main.py)**

- Loads the embeddings and initializes Elasticsearch.
- Provides an API endpoint (`/ask_question/`) for handling user queries.
- Uses different search strategies:
  - **Keyword Search** (BM25)
  - **Vector Search** (cosine similarity)
  - **Hybrid Search** (combination of keyword & vector search with reranking)

### **3. Elasticsearch Utils (elasticsearch\_utils.py)**

- Defines helper functions for different search strategies.
- Handles keyword-based, vector-based, and hybrid retrieval methods.
- Uses Elasticsearch's built-in ranking and reranking mechanisms.

### **4. LangChain Agent (agent.py)**

- Defines a **RetrievalQA chain** to fetch and process results from Elasticsearch.
- Uses **GPT-4-turbo** for additional context-aware responses.
- Implements a Pandas DataFrame agent to analyze numeric breed data.

### **5. Streamlit UI (app.py)**

- Provides a simple frontend for user interaction.
- Sends user queries to FastAPI and displays results.
- Maintains a conversation history.

### **6. Dockerized Deployment**

- The `docker-compose.yml` file defines services for **FastAPI, Elasticsearch, and Streamlit**.
- The `Dockerfile` installs required dependencies and runs the application.
- The setup ensures that the project can be deployed with a single command:
  ```sh
  docker-compose up --build
  ```

---

## Installation & Usage

### **1. Local Setup**

#### Install dependencies

```sh
pip install -r requirements.txt
```

#### Run FastAPI Backend

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Run Streamlit UI

```sh
streamlit run app.py
```

### **2. Dockerized Deployment**

#### Build and Run the Application

```sh
docker-compose up --build
```

---

## Future Enhancements

- Implement **query expansion** to improve search recall.
- Optimize embeddings storage for better retrieval performance.
- Add **multi-modal capabilities** (image-based retrieval for dog breeds).

