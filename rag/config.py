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

    # --- Découpage des documents ---
    chunk_size: int = 1000        # taille cible d'un chunk, en caractères
    chunk_overlap: int = 150      # recouvrement entre deux chunks consécutifs

    # --- Recherche ---
    top_k: int = 4                # nombre de chunks récupérés par question
    score_threshold: float = 0.0  # score minimal (similarité cosinus) pour garder un chunk

    # --- Divers ---
    data_dir: str = "data"        # dossier où sont déposés les PDF

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
            chunk_size=_get_int("CHUNK_SIZE", 1000),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
            top_k=_get_int("TOP_K", 4),
            data_dir=os.getenv("DATA_DIR", "data"),
        )

        # Garde-fou : un recouvrement >= taille de chunk provoquerait une boucle infinie
        # dans le découpage.
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP doit être strictement inférieur à CHUNK_SIZE.")

        return settings
