# rag_wowinfo/schemas.py
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator

class Source(BaseModel):
    document: str
    metadata: Dict[str, str]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

class Feedback(BaseModel):
    query_id: str
    feedback: str
    comment: Optional[str] = None  # Optional detailed feedback

class DocumentUpload(BaseModel):  # For /context
    content: str

class DocumentSummaryRequest(BaseModel): # For /summarize
    document_id: Optional[str] = None
    document_text: Optional[str] = None
    urls: Optional[List[str]] = None
    summary_length: str = "medium"
    summary_style: str = "general"

    @field_validator("document_id", "document_text", "urls", mode="after")
    def check_input_provided(cls, field_value, field):
        values = field_value
        if not any([values]):
          raise ValueError("You must provide either document_id, document_text, or urls")
        return field_value

class DocumentComparisonRequest(BaseModel): # For /compare
    document1_id: Optional[str] = None
    document1_text: Optional[str] = None
    document2_id: Optional[str] = None
    document2_text: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class MultiTurnRequest(BaseModel):
    query: str
    session_id: str

class GeneratedQuestionsRequest(BaseModel):
    document_id: Optional[str] = None
    document_text: Optional[str] = None
    num_questions: int = 5

class ParaphraseRequest(BaseModel):
  text: str

class ExtractedEntities(BaseModel):
  entity: str
  type: str

class NERResponse(BaseModel):
  entities: List[ExtractedEntities]
