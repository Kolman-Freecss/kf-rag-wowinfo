import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

def main():
    # Load environment variables
    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    chroma_host = os.environ.get("CHROMA_HOST", "localhost")

    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-001')

    # Configure ChromaDB
    client = chromadb.HttpClient(host=chroma_host, port=8000)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection("wowinfo", embedding_function=embedding_function)

    # Load data from CSV
    df = pd.read_csv("data/wow_data.csv")
    documents = df['description'].tolist()
    metadatas = df[['class', 'spec']].to_dict('records')
    ids = [f"{row['class']}-{row['spec']}" for index, row in df.iterrows()]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    # RAG logic
    query = "What is the best class in World of Warcraft?"
    results = collection.query(query_texts=[query], n_results=5)
    context = results['documents'][0]

    prompt = f"Answer the following question based on this context: {context}. Question: {query}"
    response = model.generate_content(prompt)

    print(response.text)

if __name__ == "__main__":
    main()
