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
    *   `/query`:  Send a GET request to `http://localhost:8000/query?query=<your_question>&num_results=<number_of_results>&creativity=<creativity_value>` to ask a question about World of Warcraft.
        *   `num_results`: (optional) The number of search results to retrieve (default: 5).
        *   `creativity`: (optional) The creativity of the response (0.0-1.0) (default: 0.5).
    *   `/feedback`: Send a POST request to `http://localhost:8000/feedback` with `query_id` and `feedback` in the request body to provide feedback on the answers.
    *   `/load_data`: Send a POST request to `http://localhost:8000/load_data` to reload the data from the CSV file into the ChromaDB collection.

## API Endpoints

* `/query`: Send a GET request to `http://localhost:8000/query?query=<your_question>&num_results=<number_of_results>&creativity=<creativity_value>&max_length=<max_length>&response_format=<response_format>&additional_context=<additional_context>` to ask a question about World of Warcraft.
    * `query`: (required) The question to ask.
    * `num_results`: (optional) The number of search results to retrieve (default: 5).
    * `creativity`: (optional) The creativity of the response (0.0-1.0) (default: 0.5).
    * `max_length`: (optional) The maximum length of the response.
    * `response_format`: (optional) The format of the response.
    * `additional_context`: (optional) Additional context to provide to the model.
* `/feedback`: Send a POST request to `http://localhost:8000/feedback` with `query_id` and `feedback` in the request body to provide feedback on the answers.
* `/summarize`: Send a POST request to `http://localhost:8000/summarize` with `document_id`, `document_text`, `urls`, `summary_length`, and `summary_style` in the request body to summarize a document.
    * `document_id`: (optional) The ID of the document to summarize.
    * `document_text`: (optional) The text of the document to summarize.
    * `urls`: (optional) A list of URLs to summarize.
    * `summary_length`: (optional) The length of the summary (default: medium).
    * `summary_style`: (optional) The style of the summary (default: general).
* `/compare`: Send a POST request to `http://localhost:8000/compare` with `document1_id`, `document1_text`, `document2_id`, and `document2_text` in the request body to compare two documents.
    * `document1_id`: (optional) The ID of the first document to compare.
    * `document1_text`: (optional) The text of the first document to compare.
    * `document2_id`: (optional) The ID of the second document to compare.
    * `document2_text`: (optional) The text of the second document to compare.
* `/translate`: Send a POST request to `http://localhost:8000/translate` with `text` and `target_language` in the request body to translate a text.
    * `text`: (required) The text to translate.
    * `target_language`: (required) The target language.
* `/multi_turn`: Send a POST request to `http://localhost:8000/multi_turn` with `query` and `session_id` in the request body to handle multi-turn conversations.
    * `query`: (required) The question to ask.
    * `session_id`: (required) The session ID.
* `/new_session`: Send a GET request to `http://localhost:8000/new_session` to create a new session ID for multi-turn conversations.
* `/generate_questions`: Send a POST request to `http://localhost:8000/generate_questions` with `document_id`, `document_text`, and `num_questions` in the request body to generate questions based on a document.
    * `document_id`: (optional) The ID of the document to generate questions from.
    * `document_text`: (optional) The text of the document to generate questions from.
    * `num_questions`: (optional) The number of questions to generate (default: 5).
* `/paraphrase`: Send a POST request to `http://localhost:8000/paraphrase` with `text` in the request body to paraphrase a text.
    * `text`: (required) The text to paraphrase.
* `/extract_entities`: Send a POST request to `http://localhost:8000/extract_entities` with `text` in the request body to extract entities from a text.
    * `text`: (required) The text to extract entities from.
* `/admin/add_document`: Send a POST request to `http://localhost:8000/admin/add_document` with `document`, `metadata`, and `doc_id` in the request body to add a document to the knowledge base. Requires authentication.
    * `document`: (required) The document to add.
    * `metadata`: (required) The metadata of the document.
    * `doc_id`: (required) The ID of the document.
* `/admin/update_document`: Send a POST request to `http://localhost:8000/admin/update_document` with `doc_id`, `document`, and `metadata` in the request body to update a document in the knowledge base. Requires authentication.
    * `doc_id`: (required) The ID of the document to update.
    * `document`: (optional) The updated document.
    * `metadata`: (optional) The updated metadata.
* `/admin/delete_document`: Send a DELETE request to `http://localhost:8000/admin/delete_document?doc_id=<doc_id>` to delete a document from the knowledge base. Requires authentication.
    * `doc_id`: (required) The ID of the document to delete.
* `/admin/reload_data`: Send a POST request to `http://localhost:8000/admin/reload_data` to reload the data from the CSV file into the ChromaDB collection. Requires authentication.

## Docs

- https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models?hl=es-419
