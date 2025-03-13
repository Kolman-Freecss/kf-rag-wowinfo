# KF RAG WoW Information

A Retrieval-Augmented Generation (RAG) application for World of Warcraft information. This project enhances large language model responses with accurate and up-to-date WoW game information.

## Overview

This application uses RAG architecture to:
- Index and store World of Warcraft information in a vector database (ChromaDB)
- Retrieve relevant context based on user queries
- Generate accurate responses by augmenting LLM (Gemini) with retrieved information
- Provide specific, factual answers about WoW lore, mechanics, classes, raids, etc.

## Technology Stack

- Python
- ChromaDB for vector storage
- Google Gemini API
- Docker & Docker Compose
- LangChain for RAG implementation

## Setup

1.  Install Docker and Docker Compose.
2.  Clone this repository.
3.  Create a `.env` file in the root directory with the `GEMINI_API_KEY` environment variable. You can use the `.env.template` file as a template.
4.  Run `docker-compose up --build` to start the ChromaDB and RAG application.

## Usage

1.  Run `docker-compose up --build` to start the ChromaDB and RAG application.
2.  Access the API endpoints:
    *   `/query`:  Send a GET request to `http://localhost:8000/query?query=<your_question>` to ask a question about World of Warcraft.
    *   `/load_data`: Send a POST request to `http://localhost:8000/load_data` to reload the data from the CSV file into the ChromaDB collection.

## Docs

- https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models?hl=es-419