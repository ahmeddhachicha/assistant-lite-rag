"""Configuration centrale du pipeline Lite RAG.

Toutes les valeurs sont surchargeables par variables d'environnement
(voir .env.example).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Modèle LLM servi par Ollama.
# qwen2.5:1.5b est préféré à qwen2.5:0.5b : lors de nos tests, le 0.5b
# hallucine sur les questions absentes du corpus, le 1.5b les refuse
# correctement tout en restant léger (~1 Go).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# URL du serveur Ollama (non exposé publiquement)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Chemin du corpus documentaire
CORPUS_PATH = os.getenv("CORPUS_PATH", "corpus_de_travail.txt")

# Taille des segments (en caractères) et chevauchement
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Nombre de segments récupérés lors de la recherche vectorielle
TOP_K = int(os.getenv("TOP_K", "3"))

# Timeout de l'appel LLM (secondes)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

# Fichier de suivi MLOps (une ligne JSON par requête)
LOG_FILE = os.getenv("LOG_FILE", "logs/queries.jsonl")
