# rag_wowinfo/database.py
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()  #Loads .env *before* using os.environ
chroma_host = os.environ.get("CHROMA_HOST", "localhost")
print(f"CHROMA_HOST: {chroma_host}")

client = chromadb.HttpClient(host=chroma_host, port=8000)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_collection(collection_name="wowinfo"):
    return client.get_or_create_collection(collection_name, embedding_function=embedding_function)

def load_data_to_chroma(csv_path="data/wow_data.csv", collection_name="wowinfo"):
    collection = get_collection(collection_name)
    df = pd.read_csv(csv_path)
    documents = df['description'].tolist()
    metadatas = df[['class', 'spec']].to_dict('records')
    ids = [f"{row['class']}-{row['spec']}" for index, row in df.iterrows()]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

def query_chroma(collection, query_texts, n_results=5):
    return collection.query(query_texts=query_texts, n_results=n_results)

def add_document_to_chroma(collection, document, metadata, doc_id):
    collection.add(documents=[document], metadatas=[metadata], ids=[doc_id])

def update_document_in_chroma(collection, doc_id, document=None, metadata=None):
    update_args = {}
    if document:
        update_args['documents'] = [document]
    if metadata:
        update_args['metadatas'] = [metadata]
    collection.update(ids=[doc_id], **update_args)

def delete_document_from_chroma(collection, doc_id):
    collection.delete(ids=[doc_id])

def get_document_by_id(collection, doc_id):
    result = collection.get(ids=[doc_id], include=["documents", "metadatas"])
    if result and result['documents']: # Check that there are results
      return {
        "document": result['documents'][0],
        "metadata": result['metadatas'][0] if result['metadatas'] else None
      }
    else:
      return None

#Collection initialization (optional, you can do it in a separate script)
# load_data_to_chroma() #Uncomment to load initial data
