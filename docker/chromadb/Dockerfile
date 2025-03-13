FROM python:3.9-slim-buster

WORKDIR /app

RUN pip install --target /app chromadb sentence-transformers google-generativeai python-dotenv pandas

ENV PATH="/app:${PATH}"

EXPOSE 8000

CMD ["chromadb", "--path", "/app/chroma_db"]
