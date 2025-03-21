FROM --platform=linux/arm64 python:3.9-slim

WORKDIR /code

# Create database directory
RUN mkdir -p /code/database

# Copy database files
COPY ./database/pipeline.pkl /code/database/pipeline.pkl
COPY ./database/database.json /code/database/database.json
COPY ./database/triples.json /code/database/triples.json

# Copy code files
COPY requirements.txt /code/requirements.txt
COPY setup.py /code/setup.py
COPY knowledge_database /code/knowledge_database
COPY api /code/api

# Install build tools
RUN apt-get update && apt-get install -y build-essential

# Install dependencies with compatible versions for ARM64
RUN pip install --upgrade pip && \
    pip install numpy==1.24.3 && \
    pip install pandas==1.5.3 && \
    pip install scikit-learn==1.2.2 && \
    pip install networkx rdflib openai fastapi uvicorn && \
    pip install cherche lenlp && \
    pip install --no-cache-dir .

# Set up the API key
RUN --mount=type=secret,id=OPENAI_API_KEY \
    sh -c 'echo "export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY)" >> /etc/profile.d/openai.sh'

CMD ["/bin/bash", "-c", "source /etc/profile && source /etc/profile.d/openai.sh && uvicorn api.api:app --host 0.0.0.0 --port 8080"]