"""Étape 4 du pipeline : le stockage et la recherche vectorielle (Qdrant).

Qdrant stocke des "points". Chaque point contient :
  - un identifiant,
  - un vecteur (l'embedding du chunk),
  - un `payload` : les métadonnées, ici le texte du chunk et sa provenance.

C'est ce payload qui permet, après la recherche, de reconstruire le chunk et
d'afficher la référence (document + lignes) à l'utilisateur.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# Espace de noms fixe pour générer des identifiants déterministes.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class VectorStore(ABC):
    """Contrat commun à toutes les bases vectorielles."""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Insère (ou met à jour) des chunks et retourne le nombre de points écrits."""

    @abstractmethod
    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Retourne les `top_k` chunks les plus proches du vecteur fourni."""


class QdrantVectorStore(VectorStore):
    """Implémentation Qdrant avec similarité cosinus."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
        score_threshold: float = 0.0,
    ) -> None:
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._vector_size = vector_size
        self._score_threshold = score_threshold
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Cycle de vie de la collection
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """Crée la collection si elle n'existe pas encore (opération idempotente)."""
        try:
            exists = self._client.collection_exists(self._collection)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                f"Impossible de joindre Qdrant. Le serveur est-il démarré "
                f"(`docker compose up -d`) ? Détail : {error}"
            ) from error

        if exists:
            info = self._client.get_collection(self._collection)
            configured = info.config.params.vectors.size
            if configured != self._vector_size:
                raise RuntimeError(
                    f"La collection '{self._collection}' attend des vecteurs de taille "
                    f"{configured}, or le modèle en produit {self._vector_size}. "
                    "Supprimez la collection ou changez QDRANT_COLLECTION."
                )
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )
        logger.info("Collection Qdrant '%s' créée.", self._collection)

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Le nombre de chunks et de vecteurs doit être identique.")
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=self._point_id(chunk),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "index": chunk.index,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        # `upsert` = insertion ou mise à jour. Combiné à un identifiant
        # déterministe, ré-indexer deux fois le même PDF ne crée pas de doublons.
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.info("%d point(s) écrit(s) dans '%s'.", len(points), self._collection)
        return len(points)

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        """Identifiant stable dérivé de la provenance du chunk."""
        key = f"{chunk.source}:{chunk.index}"
        return str(uuid.uuid5(_ID_NAMESPACE, key))

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------
    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            score_threshold=self._score_threshold or None,
        )

        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                source=payload.get("source", "inconnu"),
                index=int(payload.get("index", 0)),
                text=payload.get("text", ""),
                line_start=int(payload.get("line_start", 0)),
                line_end=int(payload.get("line_end", 0)),
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(point.score)))

        logger.info("Recherche : %d chunk(s) retrouvé(s).", len(results))
        return results

    # ------------------------------------------------------------------
    # Administration (pratique depuis l'interface)
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Nombre total de chunks indexés."""
        return self._client.count(self._collection, exact=True).count

    def list_sources(self) -> list[str]:
        """Liste les noms de documents présents dans la collection."""
        sources: set[str] = set()
        offset = None

        # `scroll` pagine sur l'ensemble des points ; suffisant à petite échelle.
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for point in points:
                if point.payload and point.payload.get("source"):
                    sources.add(point.payload["source"])
            if offset is None:
                break

        return sorted(sources)

    def delete_source(self, source: str) -> None:
        """Supprime tous les chunks provenant d'un document donné."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            wait=True,
        )
        logger.info("Document '%s' supprimé de la collection.", source)

    def reset(self) -> None:
        """Vide complètement la collection (utile en développement)."""
        self._client.delete_collection(self._collection)
        self._ensure_collection()
        logger.info("Collection '%s' réinitialisée.", self._collection)
