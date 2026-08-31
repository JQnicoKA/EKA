"""Étape 3 du pipeline : la vectorisation (embeddings).

Un embedding est un vecteur de nombres qui représente le *sens* d'un texte.
Deux textes proches sémantiquement ont des vecteurs proches géométriquement :
c'est ce qui permet la recherche par similarité dans Qdrant.

Point important : **le même modèle doit être utilisé pour les documents et pour
la question**, sinon les vecteurs ne sont pas comparables.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from mistralai import Mistral

from .utils import retry

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Contrat commun à tous les modèles d'embedding."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de textes (documents à indexer)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Vectorise une question utilisateur."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Taille des vecteurs produits."""


class MistralEmbedder(Embedder):
    """Implémentation basée sur l'API Mistral (`mistral-embed`, 1024 dimensions)."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed",
        dimension: int = 1024,
        batch_size: int = 32,
    ) -> None:
        self._client = Mistral(api_key=api_key)
        self._model = model
        self._dimension = dimension
        # L'API accepte plusieurs textes par requête : on regroupe pour limiter
        # le nombre d'appels réseau (et donc le risque de rate limit).
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
            logger.info("Embeddings : %d/%d chunk(s)", len(vectors), len(texts))

        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    # ------------------------------------------------------------------
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Appelle l'API Mistral pour un lot de textes, avec réessais."""
        response = retry(
            lambda: self._client.embeddings.create(model=self._model, inputs=batch),
            description="l'appel aux embeddings Mistral",
        )

        vectors = [item.embedding for item in response.data]

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"L'API a renvoyé {len(vectors)} vecteur(s) pour {len(batch)} texte(s)."
            )

        # Vérification de cohérence : la collection Qdrant est créée avec une
        # taille de vecteur fixe, un écart ici provoquerait une erreur obscure
        # au moment de l'insertion.
        actual = len(vectors[0])
        if actual != self._dimension:
            raise RuntimeError(
                f"Dimension inattendue : {actual} au lieu de {self._dimension}. "
                "Ajustez EMBEDDING_DIMENSION et recréez la collection Qdrant."
            )

        return vectors
