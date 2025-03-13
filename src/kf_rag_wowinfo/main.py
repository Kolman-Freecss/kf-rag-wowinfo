import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

def answer_question(collection, model, query):
    results = collection.query(query_texts=[query], n_results=5)
    context = results['documents'][0]
    if results['metadatas']:
        # Access the first metadata dictionary in the list
        metadata = results['metadatas'][0][0]  
        prompt = f"Answer the following question based on this context: {context}. Also, identify the class and spec from the context. Question: {query}"
        response = model.generate_content(prompt)
        return f"{response.text} Class: {metadata['class']}, Spec: {metadata['spec']}"
    else:
        return "No information found for this question."

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

    # Questions
    question1 = "What is the best class in World of Warcraft?"
    question2 = "What is a melee fighter who uses stealth and poisons to deal damage?"
    question3 = "Who is the class and spec that uses magic to heal and protect their allies? and tell me their class and spec"

    # Answer questions
    answer1 = answer_question(collection, model, question1)
    answer2 = answer_question(collection, model, question2)
    answer3 = answer_question(collection, model, question3)

    # Print answers
    print(f"Question 1: {question1}")
    print(answer1)
    print(f"Question 2: {question2}")
    print(answer2)
    print(f"Question 3: {question3}")
    print(answer3)

if __name__ == "__main__":
    main()