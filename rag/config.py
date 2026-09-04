"""Configuration centralisée de l'application.

Toute la configuration est lue depuis les variables d'environnement (fichier `.env`).
On utilise une dataclass *frozen* (immuable) : la configuration est chargée une fois
au démarrage puis ne change plus, ce qui évite les effets de bord.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Charge le fichier .env s'il existe (ne fait rien sinon).
load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Lit une variable d'environnement entière, avec valeur par défaut."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"La variable d'environnement {name}='{raw}' n'est pas un entier.") from exc


@dataclass(frozen=True)
class Settings:
    """Paramètres de l'application RAG."""

    # --- Mistral (embeddings + génération) ---
    mistral_api_key: str
    embedding_model: str = "mistral-embed"
    # Dimension des vecteurs produits par `mistral-embed`. Doit correspondre
    # à la taille déclarée dans la collection Qdrant.
    embedding_dimension: int = 1024
    llm_model: str = "mistral-small-latest"

    # --- Qdrant (base vectorielle) ---
    # 6343 est le port exposé par le docker-compose de ce projet.
    qdrant_url: str = "http://localhost:6343"
    qdrant_api_key: str | None = None
    collection_name: str = "documents"

    # --- Découpage sémantique des documents ---
    # "semantic" : le LLM planifie les frontières ; "heuristic" : plan déterministe
    # (sans appel API, utile pour l'ingestion en masse).
    chunking_mode: str = "semantic"
    # Qui planifie le découpage : "mistral" (crédits API) ou "ollama" (modèle local).
    planner_provider: str = "mistral"
    planner_model: str = ""            # vide = llm_model (mistral) ou ollama_model
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_num_ctx: int = 8192         # fenêtre de contexte du modèle local
    ollama_timeout: float = 180.0      # un modèle local peut être lent
    target_chunk_chars: int = 1200     # taille visée d'un chunk
    max_chunk_chars: int = 2000        # taille maximale (garantie par le code)
    min_chunk_chars: int = 200         # en deçà, le chunk est fusionné avec le suivant
    max_block_chars: int = 1200        # au-delà, un bloc est redécoupé par le parser
    planner_batch_blocks: int = 40     # nombre de blocs envoyés au LLM par appel
    planner_max_attempts: int = 2      # tentatives avant repli déterministe

    # --- Recherche ---
    top_k: int = 4                # nombre de chunks récupérés par question
    score_threshold: float = 0.0  # score minimal (similarité cosinus) pour garder un chunk

    # --- Divers ---
    data_dir: str = "data"        # dossier où sont déposés les .txt

    @classmethod
    def from_env(cls) -> "Settings":
        """Construit les paramètres depuis l'environnement et valide le minimum vital."""
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY est manquante. "
                "Copiez `.env.example` vers `.env` et renseignez votre clé API Mistral."
            )

        settings = cls(
            mistral_api_key=api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL", "mistral-embed"),
            embedding_dimension=_get_int("EMBEDDING_DIMENSION", 1024),
            llm_model=os.getenv("LLM_MODEL", "mistral-small-latest"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6343"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
            chunking_mode=os.getenv("CHUNKING_MODE", "semantic").strip().lower(),
            planner_provider=os.getenv("PLANNER_PROVIDER", "mistral").strip().lower(),
            planner_model=os.getenv("PLANNER_MODEL", "").strip(),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").strip(),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip(),
            ollama_num_ctx=_get_int("OLLAMA_NUM_CTX", 8192),
            ollama_timeout=float(_get_int("OLLAMA_TIMEOUT", 180)),
            target_chunk_chars=_get_int("TARGET_CHUNK_CHARS", 1200),
            max_chunk_chars=_get_int("MAX_CHUNK_CHARS", 2000),
            min_chunk_chars=_get_int("MIN_CHUNK_CHARS", 200),
            max_block_chars=_get_int("MAX_BLOCK_CHARS", 1200),
            planner_batch_blocks=_get_int("PLANNER_BATCH_BLOCKS", 40),
            planner_max_attempts=_get_int("PLANNER_MAX_ATTEMPTS", 2),
            top_k=_get_int("TOP_K", 4),
            data_dir=os.getenv("DATA_DIR", "data"),
        )

        # Garde-fous : des tailles incohérentes rendraient le découpage impossible.
        if settings.chunking_mode not in ("semantic", "heuristic"):
            raise ValueError("CHUNKING_MODE doit valoir 'semantic' ou 'heuristic'.")
        if settings.planner_provider not in ("mistral", "ollama"):
            raise ValueError("PLANNER_PROVIDER doit valoir 'mistral' ou 'ollama'.")
        if settings.target_chunk_chars > settings.max_chunk_chars:
            raise ValueError("TARGET_CHUNK_CHARS doit être <= MAX_CHUNK_CHARS.")
        if settings.max_block_chars > settings.max_chunk_chars:
            raise ValueError(
                "MAX_BLOCK_CHARS doit être <= MAX_CHUNK_CHARS : un bloc doit toujours "
                "tenir dans un chunk."
            )
        if settings.min_chunk_chars >= settings.max_chunk_chars:
            raise ValueError("MIN_CHUNK_CHARS doit être < MAX_CHUNK_CHARS.")

        return settings
