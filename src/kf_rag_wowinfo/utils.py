# rag_wowinfo/utils.py
import re
from typing import List
from urllib.parse import urlparse
import httpx

def clean_text(text: str) -> str:
    """Cleans the text: removes extra spaces, special characters, etc."""
    text = re.sub(r'\s+', ' ', text).strip()  # Espacios extra
    text = re.sub(r'[^\w\s]', '', text)  # Caracteres no alfanuméricos (opcional)
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divides the text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


async def get_url_content(url):
    """
    Gets the content of a URL. Basic implementation,
    handles errors and uses httpx to be asynchronous.
    """
    try:
        async with httpx.AsyncClient() as client: # Uses httpx for async requests
            response = await client.get(url, follow_redirects=True, timeout=10) # Follow redirects
            response.raise_for_status() # Raises exception if there is an HTTP error (4xx, 5xx)
            return response.text

    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}.")
        return None
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        return None
