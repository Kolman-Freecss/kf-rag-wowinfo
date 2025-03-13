from fastapi import FastAPI, Query
from typing import Optional
from .main import answer_question, collection, model

app = FastAPI()

@app.get("/query")
async def query_endpoint(query: str = Query(..., title="Query", description="The question to ask")):
    """
    Endpoint to answer questions about World of Warcraft.
    """
    answer = answer_question(collection, model, query)
    return {"answer": answer}

@app.post("/load_data")
async def load_data_endpoint():
    """
    Endpoint to reload data from the CSV file.
    """
    from .main import load_data
    load_data()
    return {"message": "Data loaded successfully"}
