FROM python:3.9-buster

# Install system dependencies, INCLUDING a newer SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and compile SQLite (from source, for a specific version)
WORKDIR /tmp
RUN wget https://www.sqlite.org/2022/sqlite-autoconf-3390200.tar.gz \
    && tar xzf sqlite-autoconf-3390200.tar.gz \
    && cd sqlite-autoconf-3390200 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install

# Tell the dynamic linker where to find the new SQLite library
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set the working directory
WORKDIR /app

# Copy requirements file FIRST.
COPY ./requirements.txt /app/requirements.txt

# Install chromadb and dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for the database.
RUN mkdir -p /app/chroma_db

# Expose the port
EXPOSE 8000

# Use uvicorn.
CMD ["uvicorn", "chromadb.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]