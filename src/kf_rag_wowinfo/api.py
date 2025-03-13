# rag_wowinfo/api.py
from fastapi import FastAPI, Query, HTTPException, Form, Depends
from typing import Optional, List, Dict
from .schemas import QueryResponse, Feedback, DocumentUpload, DocumentSummaryRequest, DocumentComparisonRequest, TranslationRequest, MultiTurnRequest, GeneratedQuestionsRequest, ParaphraseRequest, NERResponse
from .main import answer_question, summarize_content, compare_documents, translate_with_context, multi_turn_qa, generate_questions_from_text, paraphrase_text, extract_entities_from_text
from .database import get_collection, add_document_to_chroma, update_document_in_chroma, delete_document_from_chroma, load_data_to_chroma, get_document_by_id
from .utils import clean_text, is_valid_url, get_url_content
import os
import uuid
from fastapi.openapi.utils import get_openapi


app = FastAPI()

def custom_openapi():
    """Customizes the OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom Title",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-margin-100.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    """Returns the OpenAPI schema in JSON format."""
    return app.openapi()

# Database simulation for conversation history (multi-turn)
conversation_history = {}


# --- Endpoints ---
@app.get("/query", response_model=QueryResponse)
async def query_endpoint(
    query: str = Query(..., title="Query", description="The question to ask"),
    num_results: int = Query(5, title="Number of Results", description="Number of search results"),
    creativity: float = Query(0.5, title="Creativity", ge=0.0, le=1.0),
    max_length: Optional[int] = Query(None, title="Max Length"),
    response_format: Optional[str] = Query(None, title="Response Format"),
    additional_context: Optional[str] = Query(None, title="Additional Context")
):
    """Answers questions using the RAG model.

    Args:
        query (str): The question to ask.
        num_results (int): The number of search results to retrieve.
        creativity (float): The creativity of the response (0.0-1.0).
        max_length (Optional[int]): The maximum length of the response.
        response_format (Optional[str]): The format of the response.
        additional_context (Optional[str]): Additional context to provide to the model.

    Returns:
        QueryResponse: The answer and sources from the RAG model.
    """
    result = answer_question("wowinfo", query, num_results, creativity, max_length, response_format, additional_context)
    return result


@app.post("/feedback", status_code=201)
async def feedback_endpoint(feedback: Feedback):
    """Receives feedback on the answers.

    Args:
        feedback (Feedback): The feedback object containing the query ID, feedback, and comment.

    Returns:
        dict: A message indicating that the feedback was received successfully.
    """
    # proyecto_rag/api.py (continuación)
    """Receives feedback on the answers."""
    # In a real application, you would save this in a database
    print(f"Received feedback for query {feedback.query_id}: {feedback.feedback}, comment: {feedback.comment}")
    return {"message": "Feedback received successfully"}


@app.post("/context", status_code=201)
async def context_endpoint(upload: DocumentUpload):
    """Allows users to provide additional context for a question.

    Args:
        upload (DocumentUpload): The document upload object containing the context content.

    Returns:
        dict: A message indicating that the context was received and the cleaned context.
    """
    # You may want to do something special with the context here, like preprocessing it
    cleaned_context = clean_text(upload.content)
    return {"message": "Context received", "cleaned_context": cleaned_context}


@app.post("/summarize", response_model=str)
async def summarize_endpoint(request: DocumentSummaryRequest):
    """Summarizes a document, given either its ID, text, or a list of URLs.

    Args:
        request (DocumentSummaryRequest): The request object containing the document ID, text, URLs, summary length, and summary style.

    Returns:
        str: The summarized text.
    """
    document_text = ""

    if request.document_id:
        doc_info = get_document_by_id(get_collection(), request.document_id)
        if doc_info:
            document_text = doc_info["document"]
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    elif request.document_text:
        document_text = request.document_text

    elif request.urls:
        # Fetch content from URLs *asynchronously*
        url_contents = []
        for url in request.urls:
            if not is_valid_url(url):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
            content = await get_url_content(url)  # Usa la función asíncrona
            if content:
                url_contents.append(content)
        document_text = "\n\n".join(url_contents)


    if not document_text: #Verificación adicional
        raise HTTPException(status_code=400, detail="No document content provided")


    summary = await summarize_content(document_text, request.summary_length, request.summary_style)
    return summary

@app.post("/compare", response_model=str)
async def compare_endpoint(request: DocumentComparisonRequest):
    """Compares two documents.

    Args:
        request (DocumentComparisonRequest): The request object containing the document IDs or texts to compare.

    Returns:
        str: A comparison of the two documents.
    """
    doc1_text = ""
    doc2_text = ""

    if request.document1_id:
      doc1_info = get_document_by_id(get_collection(), request.document1_id)
      if doc1_info:
          doc1_text = doc1_info["document"]
      else:
          raise HTTPException(status_code=404, detail="Document 1 not found")
    elif request.document1_text:
        doc1_text = request.document1_text


    if request.document2_id:
      doc2_info = get_document_by_id(get_collection(), request.document2_id)
      if doc2_info:
          doc2_text = doc2_info["document"]
      else:
          raise HTTPException(status_code=404, detail="Document 2 not found")
    elif request.document2_text:
      doc2_text = request.document2_text

    if not doc1_text or not doc2_text: #Verificación adicional
      raise HTTPException(status_code=400, detail="Both document texts are required")


    comparison = await compare_documents(doc1_text, doc2_text)
    return comparison

@app.post("/translate", response_model=str)
async def translate_endpoint(request: TranslationRequest):
  """Translates a text, taking into account RAG context.

  Args:
      request (TranslationRequest): The request object containing the text to translate and the target language.

  Returns:
      str: The translated text.
  """
  translation = await translate_with_context(request.text, request.target_language)
  return translation

@app.post("/multi_turn", response_model=QueryResponse)
async def multi_turn_endpoint(request: MultiTurnRequest):
    """Handles multi-turn conversations.

    Args:
        request (MultiTurnRequest): The request object containing the query and session ID.

    Returns:
        QueryResponse: The answer and sources from the RAG model.
    """
    session_id = request.session_id
    if session_id not in conversation_history:
        conversation_history[session_id] = []  # Initialize history

    result = await multi_turn_qa(request.query, session_id, conversation_history[session_id])
    return result

@app.get("/new_session")
async def new_session_endpoint():
    """Creates a new session ID for multi-turn conversations.

    Returns:
        dict: A dictionary containing the new session ID.
    """
    session_id = str(uuid.uuid4())  # Use uuid for unique IDs
    return {"session_id": session_id}


@app.post("/generate_questions", response_model=str)
async def generate_questions_endpoint(request: GeneratedQuestionsRequest):
  """Generates questions based on provided document content.

  Args:
      request (GeneratedQuestionsRequest): The request object containing the document ID, text, and number of questions to generate.

  Returns:
      str: The generated questions.
  """
  document_text = ""
  if request.document_id:
    doc_info = get_document_by_id(get_collection(), request.document_id)
    if doc_info:
        document_text = doc_info["document"]
    else:
        raise HTTPException(status_code=404, detail="Document not found")
  elif request.document_text:
      document_text = request.document_text

  if not document_text:
      raise HTTPException(status_code=400, detail="No document content provided.")

  questions = await generate_questions_from_text(document_text, request.num_questions)
  return questions

@app.post("/paraphrase", response_model=str)
async def paraphrase_endpoint(request: ParaphraseRequest):
  """Paraphrases a given text.

  Args:
      request (ParaphraseRequest): The request object containing the text to paraphrase.

  Returns:
      str: The paraphrased text.
  """
  paraphrased_text = await paraphrase_text(request.text)
  return paraphrased_text


@app.post("/extract_entities", response_model=NERResponse)
async def ner_endpoint(text: str = Form(...)):
    """Extracts named entities from a given text.

    Args:
        text (str): The text to extract entities from.

    Returns:
        NERResponse: A list of dictionaries, where each dictionary contains an entity and its type.
    """
    entities = await extract_entities_from_text(text)
    return {"entities": entities}


# --- Admin Endpoints (con autenticación básica) ---
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticates the user using HTTP Basic authentication.

    Args:
        credentials (HTTPBasicCredentials): The username and password provided in the request.

    Returns:
        str: The username if authentication is successful.

    Raises:
        HTTPException: If the username or password is incorrect.
    """
    correct_username = os.environ.get("ADMIN_USERNAME", "admin") #Usa variables de entorno
    correct_password = os.environ.get("ADMIN_PASSWORD", "password")

    if credentials.username == correct_username and credentials.password == correct_password:
        return credentials.username
    else:
      raise HTTPException(
          status_code=401,
          detail="Incorrect email or password",
          headers={"WWW-Authenticate": "Basic"},
      )


@app.post("/admin/add_document", status_code=201)
async def add_document_endpoint(
    document: str = Form(...),
    metadata: str = Form(...),
    doc_id: str = Form(...),
    username: str = Depends(get_current_username) #Autenticación
):
    """Adds a document to the knowledge base.

    Args:
        document (str): The document to add.
        metadata (str): The metadata associated with the document.
        doc_id (str): The ID of the document.
        username (str): The username of the authenticated user.

    Returns:
        dict: A message indicating that the document was added successfully.
    """
    try:
      metadata_dict = eval(metadata)  # Use eval() safely with a dictionary string.
      if not isinstance(metadata_dict, dict):
          raise ValueError("Metadata must be a dictionary")
    except (SyntaxError, ValueError) as e:
      raise HTTPException(status_code=400, detail=f"Invalid metadata format: {e}")

    add_document_to_chroma(get_collection(), document, metadata_dict, doc_id)
    return {"message": "Document added successfully"}

@app.post("/admin/update_document")
async def update_document_endpoint(
    doc_id: str = Form(...),
    document: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    username: str = Depends(get_current_username)
):
    """Updates an existing document in the knowledge base.

    Args:
        doc_id (str): The ID of the document to update.
        document (Optional[str]): The updated document content.
        metadata (Optional[str]): The updated metadata associated with the document.
        username (str): The username of the authenticated user.

    Returns:
        dict: A message indicating that the document was updated successfully.
    """
    metadata_dict = None
    if metadata:
        try:
            metadata_dict = eval(metadata)
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a dictionary")
        except (SyntaxError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata format: {e}")

    update_document_in_chroma(get_collection(), doc_id, document, metadata_dict)
    return {"message": "Document updated successfully"}

@app.delete("/admin/delete_document")
async def delete_document_endpoint(doc_id: str = Query(...), username: str = Depends(get_current_username)):
    """Deletes a document from the knowledge base.

    Args:
        doc_id (str): The ID of the document to delete.
        username (str): The username of the authenticated user.

    Returns:
        dict: A message indicating that the document was deleted successfully.
    """
    delete_document_from_chroma(get_collection(), doc_id)
    return {"message": "Document deleted successfully"}


@app.post("/admin/reload_data", status_code=201) #To reload data from the CSV
async def reload_data_endpoint(username: str = Depends(get_current_username)):
    """Reloads data from the CSV file into the ChromaDB collection.

    Args:
        username (str): The username of the authenticated user.

    Returns:
        dict: A message indicating that the data was reloaded successfully.
    """
    load_data_to_chroma()
    return {"message": "Data reloaded successfully from CSV"}
