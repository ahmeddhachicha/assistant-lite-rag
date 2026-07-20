"""Suivi MLOps simplifié : une ligne JSON par requête."""

import json
import os
from datetime import datetime, timezone

from rag import config


def log_query(
    question: str,
    model: str,
    elapsed_seconds: float,
    n_segments: int,
    status: str,
) -> None:
    """Enregistre le modèle, le temps de traitement, le nombre de
    segments récupérés et le statut de la requête."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "model": model,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "n_segments": n_segments,
        "status": status,
    }
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
