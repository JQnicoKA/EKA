"""Test de bout en bout du pipeline SANS appeler l'API Mistral.

On remplace l'embedder et le générateur par de faux composants déterministes :
cela valide toute la plomberie (.txt -> chunks -> Qdrant -> recherche -> prompt)
sans consommer de crédits API. À supprimer une fois le projet en main.

Lancement :  python smoke_test.py
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.generator import AnswerGenerator, MistralGenerator
from rag.loader import TextLoader
from rag.models import RetrievedChunk
from rag.pipeline import RAGPipeline
from rag.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DIMENSION = 64


class FakeEmbedder(Embedder):
    """Embedder factice : vecteur "sac de mots" hashé, déterministe."""

    @property
    def dimension(self) -> int:
        return DIMENSION

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIMENSION
        for word in text.lower().split():
            digest = hashlib.md5(word.encode()).digest()
            vector[digest[0] % DIMENSION] += 1.0
        return vector or [0.0] * DIMENSION


class FakeGenerator(AnswerGenerator):
    """Générateur factice : affiche le prompt qui aurait été envoyé au LLM."""

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        prompt = MistralGenerator._format_context(contexts)
        return f"[FAUSSE RÉPONSE]\nPrompt qui serait envoyé au LLM :\n\n{prompt[:600]}"


def main() -> None:
    pipeline = RAGPipeline(
        loader=TextLoader(),
        chunker=TextChunker(chunk_size=300, chunk_overlap=50),
        embedder=FakeEmbedder(),
        vector_store=QdrantVectorStore(
            url="http://localhost:6343",
            collection_name="smoke_test",
            vector_size=DIMENSION,
        ),
        generator=FakeGenerator(),
        top_k=2,
    )

    print("\n=== 1. Ingestion ===")
    nb = pipeline.ingest_file(Path("data/test_rapport.txt"))
    print(f"chunks indexés : {nb}")

    print("\n=== 2. État de la base ===")
    print("documents :", pipeline.vector_store.list_sources())
    print("chunks    :", pipeline.vector_store.count())

    print("\n=== 3. Idempotence (ré-ingestion) ===")
    pipeline.ingest_file(Path("data/test_rapport.txt"))
    print("chunks après ré-ingestion :", pipeline.vector_store.count(), "(doit être identique)")

    print("\n=== 4. Question ===")
    answer = pipeline.ask("Quel est le chiffre d affaires ?")
    print(answer.text)
    print("\nSources :")
    for i, result in enumerate(answer.sources, start=1):
        print(f"  [{i}] {result.chunk.reference} — score {result.score:.3f}")

    print("\n=== 5. Nettoyage ===")
    pipeline.vector_store.delete_source("test_rapport.txt")
    print("chunks après suppression :", pipeline.vector_store.count())
    pipeline.vector_store._client.delete_collection("smoke_test")
    print("collection de test supprimée.")

    print("\n✅ Pipeline fonctionnel de bout en bout.")


if __name__ == "__main__":
    main()
