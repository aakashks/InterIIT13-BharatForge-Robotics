# Centralized Intelligence for Dynamic Swarm Navigation

This repository provides **centralized control and object retrieval** logic for a swarm of robots operating in dynamic, GPS-denied environments. The high-level process combines environment mapping, continuous patrolling, and on-demand task assignment, ensuring robust performance in complex, ever-changing settings.

## Table of Contents
- [Centralized Intelligence for Dynamic Swarm Navigation](#centralized-intelligence-for-dynamic-swarm-navigation)
  - [Table of Contents](#table-of-contents)
  - [Overview of Workflow](#overview-of-workflow)
  - [Key Features](#key-features)
  - [Method](#method)
  - [Why This Approach?](#why-this-approach)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [Environment Variables](#environment-variables)
  - [Running the Services (on the Remote Server)](#running-the-services-on-the-remote-server)
    - [Chromadb Vector Database Embedding API](#chromadb-vector-database-embedding-api)
    - [vLLM Vision Language Model Server](#vllm-vision-language-model-server)
    - [Streamlit Application](#streamlit-application)

---

## Overview of Workflow

1. **Environment Mapping (Exploration)**  
   A swarm of robots collaboratively builds an initial map of the environment—identifying obstacles and understanding the boundaries.

2. **Continuous Monitoring (Patrolling)**  
   Robots patrol the environment, each updating the shared database with any new obstacles or changes in real time.

3. **User Queries (Task Assignment and Execution)**  
   When a user issues commands like “Go upstairs” or “Locate the nearest fire extinguisher,” the system dynamically identifies the best robot for the task and computes the shortest collision-free route.


## Key Features

- **Scalability**  
  Efficiently handles large swarms with minimal delays in task assignment and path planning.

- **Robustness**  
  Adapts seamlessly to frequent and unexpected changes in the environment without needing manual intervention.

- **Modular Architecture**  
  Easily interchangeable components (e.g., different vision-language models or database backends) to accommodate various hardware setups.


## Method

1. **Vector Similarity Search**  
   - [ChromaDB](https://docs.trychroma.com/guides/multimodal) vector database  
   - Uses OpenCLIP for creating multimodal embeddings  
   - Performs vector similarity search with the HNSW algorithm, which scales logarithmically while maintaining strong accuracy

2. **Molmo Vision-Language Model (VLM)**  
   - A vision-language model trained for pointing tasks with excellent generalization out of the box  
   - Utilizes an OpenAI-compatible API and can be swapped with any other multimodal LLM API (e.g., GPT-4o)

3. **Flexible and Scalable Deployment**  
   - The database (custom-built service) and VLM inference server (vLLM) can both run on a dedicated server  
   - Each service can handle multiple concurrent requests, significantly improving scalability

4. **Task Assignment Flow**  
   1. A lightweight LLM interprets the user query (e.g., “fire extinguisher”).  
   2. Perform vector similarity search to find the top matching objects.  
   3. **Molmo** confirms the requested object’s presence and location.  
   4. A robot is assigned a goal based on the target location determined.


## Why This Approach?

- **Generalization**  
  Combining **multimodal embeddings** with **vision-language reasoning** ensures robust performance across unfamiliar environments.

- **Efficiency**  
  Restricting VLM calls to only the top search results minimizes computational overhead while maintaining accuracy.

- **Scalability**  
  LLM and VLM are called only upon user queries, reducing no of inference requests. Embeddings are computed only when a new object is encountered.

---

## Prerequisites

- **Python 3.10+**
- **Minimum GPU**: 16 GB GPU memory (NVIDIA recommended)


## Setup

### Environment Variables

Create a `.env` file in the root directory of this project to store configuration variables. Make sure **not** to commit this file to version control. Populate the `.env` with the following:

```env
DB_URL=http://<db_server_ip>:8000
VLLM_URL=http://<vllm_server_ip>:8080
DATA_DIR=/path/to/local/data
OPENAI_API_KEY=your_openai_api_key
```

- `DB_URL` - URL where the Chromadb embedding API is hosted.  
- `VLLM_URL` - URL where the vLLM server is hosted.  
- `DATA_DIR` - Directory path on the local machine where data will be stored.  
- `OPENAI_API_KEY` - Your OpenAI API key for accessing language models.


## Running the Services (on the Remote Server)

Use separate terminal sessions (or a process manager such as `tmux`, `screen`, or `docker-compose`) to manage each service.

### Chromadb Vector Database Embedding API

The embedding API is a custom-built FastAPI service that calculates and stores image embeddings, offering an API to query images based on these embeddings.

**Start the Embedding API Server:**

```bash
cd /path/to/remote/data
# copy db_server.py to this directory if not done earler
python db_server.py ./ --port 8000
```

- `/path/to/remote/data` is where Chromadb will store its data on the remote server.

### vLLM Vision Language Model Server

The vLLM server hosts the Vision-Language Model responsible for interpreting and generating responses based on visual input.

**Start the vLLM Server:**

```bash
vllm serve allenai/Molmo-7B-D-0924 --task generate --trust-remote-code --max-model-len 4096 --dtype bfloat16 --port 8080
```

> **Note**  
> - `allenai/Molmo-7B-D-0924` specifies the vision-language model to use.  
> - Ensure the server has at least 16 GB of GPU memory.  
> - Adjust the `--port` if needed to avoid conflicts.

### Streamlit Application

The Streamlit app provides an interactive frontend for users to ask query and give target to the robot swarm.

**Start the Streamlit App:**

```bash
streamlit run app.py
```

> **Note**  
> - By default, Streamlit runs on `http://localhost:8501`.  
> - Use the `--server.port` flag to change the port if necessary.  
> - The app communicates with both the Chromadb embedding API and the vLLM server using the URLs in the `.env` file.
