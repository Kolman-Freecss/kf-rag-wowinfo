# rag_wowinfo/main.py
import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from .database import get_collection, query_chroma, add_document_to_chroma, update_document_in_chroma, delete_document_from_chroma, get_document_by_id
from .utils import clean_text, chunk_text, is_valid_url, get_url_content
from typing import Optional, List
import httpx #To make requests to URLs asynchronously

# Load environment variables and configure Gemini
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
print(f"GEMINI_API_KEY: {gemini_api_key}")
genai.configure(api_key=gemini_api_key)
# gemini-1.5-pro-002
MODEL_NAME = 'gemini-1.5-pro-002'
model = genai.GenerativeModel(MODEL_NAME)

# --- Principal functions of RAG system ---

def answer_question(collection_name: str, query: str, num_results: int = 5, creativity: float = 0.5,
                    max_length: Optional[int] = None, response_format: Optional[str] = None,
                    additional_context: Optional[str] = None):
    """Answers a question based on retrieved context from a ChromaDB collection.

    Args:
        collection_name (str): The name of the ChromaDB collection to query.
        query (str): The question to answer.
        num_results (int, optional): The number of search results to retrieve. Defaults to 5.
        creativity (float, optional): The creativity of the response (0.0-1.0). Defaults to 0.5.
        max_length (Optional[int], optional): The maximum length of the answer. Defaults to None.
        response_format (Optional[str], optional): The desired format of the response. Defaults to None.
        additional_context (Optional[str], optional): Additional context to include in the prompt. Defaults to None.

    Returns:
        Dict: A dictionary containing the answer and a list of sources.
    """
    collection = get_collection(collection_name)
    results = query_chroma(collection, [query], n_results=num_results)

    if not results or not results['documents'] or not results['documents'][0]:
        return {"answer": "No relevant information was found.", "sources": []}

    context_list = results['documents'][0]
    metadatas = results['metadatas'][0] if results['metadatas'] else []

    # Add the context if provided
    if additional_context:
        context_list.insert(0, additional_context)

    context = " ".join(context_list)  # Join the context into a single string

    prompt = f"Answer the following question based on this context: {context}. Question: {query}"
    if response_format:
        prompt += f" Please provide the answer in the following format: {response_format}."

    response = model.generate_content(prompt, safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    })

    answer = response.text
    if max_length:
        answer = answer[:max_length]
    sources = [{"document": doc, "metadata": metadata} for doc, metadata in zip(results['documents'][0], metadatas)]
    return {"answer": answer, "sources": sources}

async def summarize_content(document_text: str, summary_length: str = "medium", summary_style: str = "general"):
    """Summarizes a given text.

    Args:
        document_text (str): The text to summarize.
        summary_length (str, optional): The desired length of the summary. Defaults to "medium".
        summary_style (str, optional): The desired style of the summary. Defaults to "general".

    Returns:
        str: The summarized text.
    """
    if not document_text:
      return "Error: No document text provided."

    prompt = f"Summarize the following text in a {summary_length} length, {summary_style} style: {document_text}"
    response = model.generate_content(prompt)
    return response.text


async def compare_documents(doc1_text: str, doc2_text: str):
    """Compares two texts.

    Args:
        doc1_text (str): The text of the first document.
        doc2_text (str): The text of the second document.

    Returns:
        str: A comparison of the two texts.
    """
    if not doc1_text or not doc2_text:
        return "Error: Both document texts are required for comparison."

    prompt = f"Compare and contrast the following two texts:\n\nText 1: {doc1_text}\n\nText 2: {doc2_text}"
    response = model.generate_content(prompt)
    return response.text

async def translate_with_context(text: str, target_language: str, num_results:int = 3):
    """Translates text, using RAG for additional context.

    Args:
        text (str): The text to translate.
        target_language (str): The target language for the translation.
        num_results (int, optional): The number of RAG results to use for context. Defaults to 3.

    Returns:
        str: The translated text.
    """
    #First, RAG retrieval.
    collection = get_collection() #We obtain the default collection
    results = query_chroma(collection, [text], n_results=num_results)

    context_list = results['documents'][0] if (results and results['documents']) else []

    context = " ".join(context_list)
    prompt = f"""Translate the following text to {target_language}, also take into account this additional context for a more accurate translation:
            Text to translate: {text}
            Context: {context}
            """

    response = model.generate_content(prompt)
    return response.text

async def multi_turn_qa(query: str, session_id: str, history: List[Dict]):
    """Handles multi-turn conversations.

    Args:
        query (str): The user's query.
        session_id (str): The ID of the conversation session.
        history (List[Dict]): A list of previous turns in the conversation.

    Returns:
        Dict: A dictionary containing the answer and a list of sources.
    """
    # Retrieves the history (simulated here, ideally from a DB)
    #  We use a 'history' list as an argument *and* simulate storage

    # 2. Formatea el historial para el prompt
    history_prompt = ""
    for turn in history:
        history_prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"

    # 3. Combines the history with the current question
    full_prompt = f"{history_prompt}User: {query}"

    # 4. Usa el prompt completo para retrieval y generación
    result = answer_question(collection_name="wowinfo", query=full_prompt, num_results=3)

    # 5.  Adds the new interaction to the history (and "saves" it in the "DB")
    history.append({"user": query, "ai": result["answer"]})

    return result


async def generate_questions_from_text(text:str, num_questions:int = 5):
    """Generates questions based on a given text.

    Args:
        text (str): The text to generate questions from.
        num_questions (int, optional): The number of questions to generate. Defaults to 5.

    Returns:
        str: The generated questions.
    """
    prompt = f"""Generate {num_questions} questions based on the following text:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

async def paraphrase_text(text: str):
    """Paraphrases a given text.

    Args:
        text (str): The text to paraphrase.

    Returns:
        str: The paraphrased text.
    """
    prompt = f"Please paraphrase the following text, while trying to maintain the original meaning: {text}"
    response = model.generate_content(prompt)
    return response.text

async def extract_entities_from_text(text: str):
    """Identifies and classifies named entities in a given text.

    Args:
        text (str): The text to extract entities from.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains an entity and its type.
    """
    prompt = f"Identify and classify the named entities in the following text: {text}"
    response = model.generate_content(prompt)
    # Parseo básico (¡esto es un ejemplo simple! Debería mejorarse)
    entities = []
    for line in response.text.split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            entity = parts[0].strip()
            entity_type = parts[1].strip()
            entities.append({"entity": entity, "type": entity_type})

    return entities



# --- Auxiliary functions (could go in utils.py) ---
def generate_session_id():
    """Generates a unique session ID (simulated).

    Returns:
        str: A unique session ID.
    """
    import uuid
    return str(uuid.uuid4())
