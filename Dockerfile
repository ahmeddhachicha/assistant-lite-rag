FROM python:3.11-slim

WORKDIR /app

# Dépendances d'abord pour profiter du cache de build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, corpus et thème
COPY rag/ rag/
COPY .streamlit/ .streamlit/
COPY app.py corpus_de_travail.txt ./

# Ollama tourne hors du conteneur (Codespace / hôte)
ENV OLLAMA_URL=http://host.docker.internal:11434

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
