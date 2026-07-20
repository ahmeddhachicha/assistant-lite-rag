# Assistant documentaire Lite RAG

Application IA qui répond à des questions en langage naturel à partir d'un
corpus documentaire (discours d'investiture présidentielle, 2013), en
s'appuyant uniquement sur les passages retrouvés dans ce corpus.

**Examen fil rouge — Module MLOps, Docker et Kubernetes — École Hexagone**

## Architecture

```
Utilisateur
    ↓
Webapp Streamlit (app.py)
    ↓
Pipeline Lite RAG (rag/pipeline.py)
   ↙            ↘
Base vectorielle    Ollama + LLM léger
ChromaDB locale     (qwen2.5:1.5b)
    ↓
Corpus fourni (corpus_de_travail.txt)
```

| Composant | Fichier | Rôle |
|---|---|---|
| Interface | `app.py` | Webapp Streamlit |
| Configuration | `rag/config.py` | Paramètres (env. surchargeable) |
| Chargement / découpage | `rag/loader.py` | Lecture du corpus, segmentation |
| Recherche vectorielle | `rag/vectorstore.py` | ChromaDB local, embeddings |
| Appel LLM | `rag/llm.py` | API Ollama, prompt anti-hallucination |
| Orchestration | `rag/pipeline.py` | Chaîne complète + métriques |
| Suivi MLOps | `rag/monitoring.py` | Log JSONL par requête |

## Installation

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve            # dans un premier terminal
ollama pull qwen2.5:1.5b   # dans un second terminal

# 2. Dépendances Python
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Configuration (optionnel)
cp .env.example .env
```

## Lancement

```bash
streamlit run app.py
```

L'application est disponible sur http://localhost:8501.

### Avec Docker

```bash
docker build -t lite-rag .
docker run -p 8501:8501 --add-host=host.docker.internal:host-gateway lite-rag
```

Ollama reste exécuté sur l'hôte (Codespace) ; le conteneur le joint via
`host.docker.internal:11434`. Seul le port 8501 (Streamlit) est exposé,
le port Ollama (11434) n'est jamais rendu public.

## Choix du modèle

`qwen2.5:1.5b` (~1 Go) au lieu du `qwen2.5:0.5b` recommandé : lors de nos
tests, le 0.5b **hallucine** sur les questions absentes du corpus (il a
inventé un prix du bitcoin), tandis que le 1.5b répond correctement aux
questions présentes **et** signale correctement les informations absentes.
Il reste compatible avec la RAM d'un Codespace. Le modèle est configurable
via la variable `OLLAMA_MODEL` ou la barre latérale de l'application.

## Principaux paramètres (`.env` / barre latérale)

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Modèle LLM Ollama |
| `OLLAMA_URL` | `http://localhost:11434` | URL du serveur Ollama |
| `CHUNK_SIZE` | `800` | Taille des segments (caractères) |
| `CHUNK_OVERLAP` | `100` | Chevauchement entre segments |
| `TOP_K` | `3` | Nombre de segments récupérés |

## Suivi MLOps

Chaque requête est enregistrée dans `logs/queries.jsonl` : horodatage,
modèle, temps de traitement, nombre de segments récupérés, statut
(`ok`, `hors_corpus`, `ollama_indisponible`, `erreur`).

## Tests

```bash
python -m tests.run_tests
```

Trois questions (deux présentes dans le corpus, une absente) sur deux
configurations (nombre de segments récupérés : 3 vs 6). Résultats :
[tests/resultats_tests.md](tests/resultats_tests.md).
