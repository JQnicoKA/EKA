"""Le chef d'orchestre : assemble tous les composants du RAG.

`RAGPipeline` ne contient aucune logique métier propre ; il se contente
d'enchaîner les composants. Chaque dépendance est injectée dans le constructeur
(*dependency injection*), ce qui rend la classe testable : on peut remplacer
n'importe quel étage par un faux objet dans les tests.

Deux flux :
  - INGESTION : PDF -> pages -> chunks -> embeddings -> Qdrant
  - INTERROGATION : question -> embedding -> recherche -> prompt -> LLM -> réponse
"""

from __future__ import annotations

import logging
from pathlib import Path

from .chunker import TextChunker
from .config import Settings
from .embedder import Embedder, MistralEmbedder
from .generator import AnswerGenerator, MistralGenerator
from .loader import DocumentLoader, PDFLoader
from .models import Answer, Chunk
from .vector_store import QdrantVectorStore, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Point d'entrée unique de l'application."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: AnswerGenerator,
        top_k: int = 4,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Fabrique : construit un pipeline complet à partir de la configuration
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Assemble les implémentations concrètes (Mistral + Qdrant)."""
        embedder = MistralEmbedder(
            api_key=settings.mistral_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )

        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
            vector_size=embedder.dimension,
            score_threshold=settings.score_threshold,
        )

        return cls(
            loader=PDFLoader(),
            chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
            embedder=embedder,
            vector_store=vector_store,
            generator=MistralGenerator(
                api_key=settings.mistral_api_key,
                model=settings.llm_model,
            ),
            top_k=settings.top_k,
        )

    # ------------------------------------------------------------------
    # Flux 1 : ingestion
    # ------------------------------------------------------------------
    def ingest_file(self, path: Path) -> int:
        """Indexe un document et retourne le nombre de chunks stockés."""
        path = Path(path)
        logger.info("Ingestion de %s...", path.name)

        pages = self.loader.load(path)                       # 1. chargement
        chunks: list[Chunk] = self.chunker.split(pages)      # 2. découpage
        if not chunks:
            return 0

        vectors = self.embedder.embed_documents([c.text for c in chunks])  # 3. embeddings
        return self.vector_store.add(chunks, vectors)                      # 4. stockage

    def ingest_directory(self, directory: Path) -> dict[str, int]:
        """Indexe tous les PDF d'un dossier. Retourne {nom_fichier: nb_chunks}."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {directory}")

        results: dict[str, int] = {}
        for path in sorted(directory.glob("*.pdf")):
            try:
                results[path.name] = self.ingest_file(path)
            except Exception as error:  # noqa: BLE001
                # Un document illisible ne doit pas interrompre tout le lot.
                logger.error("Échec de l'ingestion de %s : %s", path.name, error)
                results[path.name] = 0

        return results

    # ------------------------------------------------------------------
    # Flux 2 : interrogation
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Répond à une question en s'appuyant sur les documents indexés."""
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide.")

        query_vector = self.embedder.embed_query(question)               # 5. embedding
        contexts = self.vector_store.search(query_vector, top_k or self.top_k)  # 6. recherche
        text = self.generator.generate(question, contexts)               # 7. prompt + LLM

        return Answer(question=question, text=text, sources=contexts)
