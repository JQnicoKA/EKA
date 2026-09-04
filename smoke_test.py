"""Test de bout en bout du pipeline SANS appeler l'API Mistral.

On remplace le planner, l'embedder et le générateur par de faux composants
déterministes : cela valide toute la plomberie (texte -> blocs -> plan -> validation
-> chunks -> Qdrant -> recherche -> prompt) sans consommer de crédits API.

L'étage Qdrant est ignoré si le serveur n'est pas joignable.

Lancement :  python smoke_test.py
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from rag.embedder import Embedder
from rag.generator import AnswerGenerator, MistralGenerator
from rag.models import RetrievedChunk
from rag.parser import TextParser
from rag.pipeline import RAGPipeline
from rag.planner import HeuristicChunkPlanner
from rag.semantic_chunker import SemanticChunker
from rag.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DIMENSION = 64
DOCUMENT = Path("data/test_rapport.txt")


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
        return vector


class FakeGenerator(AnswerGenerator):
    """Générateur factice : affiche le prompt qui aurait été envoyé au LLM."""

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        prompt = MistralGenerator._format_context(contexts)
        return f"[FAUSSE RÉPONSE]\nPrompt qui serait envoyé au LLM :\n\n{prompt[:600]}"


def chunking_hors_ligne() -> None:
    """Étapes 1 à 4 : parsing, plan déterministe, validation, reconstruction."""
    parser = TextParser(max_block_chars=800)
    document = parser.parse(DOCUMENT)

    print("\n=== 1. Parsing ===")
    print(f"{document.source} : {len(document.blocks)} blocs "
          f"({len(document.indexable_blocks)} indexables)")
    for block in document.blocks[:6]:
        extrait = " ".join(block.text.split())[:60]
        print(f"  {block.block_id} {block.type:10s} l.{block.line_start:<3d} « {extrait} »")

    chunker = SemanticChunker(
        planner=HeuristicChunkPlanner(target_chunk_chars=700, max_chunk_chars=1200),
        max_chunk_chars=1200,
    )
    chunks = chunker.chunk(document)

    print("\n=== 2. Plan validé + reconstruction ===")
    for chunk in chunks:
        extrait = " ".join(chunk.text.split())[:70]
        print(f"  {chunk.chunk_id} {chunk.reference:32s} {len(chunk.text):5d} car. "
              f"| {chunk.section or '(racine)'}")
        print(f"      blocs {', '.join(chunk.block_ids)} — « {extrait}... »")

    print("\n=== 3. Contrôles de fidélité ===")
    vus = [bid for chunk in chunks for bid in chunk.block_ids]
    attendus = [b.block_id for b in document.indexable_blocks]
    assert vus == attendus, "couverture ou ordre incorrects"
    for chunk in chunks:
        for block_id in chunk.block_ids:
            block = document.by_id()[block_id]
            assert block.text in chunk.text
            assert document.text[block.char_start:block.char_end] == block.text
    print(f"  couverture 100 % ({len(attendus)} blocs), ordre conservé, "
          "texte identique au document.")


def bout_en_bout() -> None:
    """Étapes 5 à 7 : stockage, recherche, prompt (nécessite Qdrant)."""
    try:
        store = QdrantVectorStore(
            url="http://localhost:6343", collection_name="smoke_test", vector_size=DIMENSION
        )
    except RuntimeError as error:
        print(f"\n=== 4. Qdrant indisponible — étage ignoré ===\n  {error}")
        return

    pipeline = RAGPipeline(
        parser=TextParser(max_block_chars=800),
        chunker=SemanticChunker(
            planner=HeuristicChunkPlanner(target_chunk_chars=700, max_chunk_chars=1200),
            max_chunk_chars=1200,
        ),
        embedder=FakeEmbedder(),
        vector_store=store,
        generator=FakeGenerator(),
        top_k=2,
    )

    print("\n=== 4. Ingestion ===")
    print("chunks indexés :", pipeline.ingest_file(DOCUMENT))
    print("documents :", store.list_sources(), "| chunks :", store.count())

    print("\n=== 5. Idempotence (ré-ingestion) ===")
    pipeline.ingest_file(DOCUMENT)
    print("chunks après ré-ingestion :", store.count(), "(doit être identique)")

    print("\n=== 6. Question ===")
    answer = pipeline.ask("Quel est le budget de la direction technique ?")
    print(answer.text)
    for position, result in enumerate(answer.sources, start=1):
        print(f"  [{position}] {result.chunk.reference} — score {result.score:.3f}")

    print("\n=== 7. Nettoyage ===")
    store.delete_source(DOCUMENT.name)
    print("chunks après suppression :", store.count())
    store._client.delete_collection("smoke_test")
    print("collection de test supprimée.")


def main() -> None:
    chunking_hors_ligne()
    bout_en_bout()
    print("\n✅ Pipeline fonctionnel.")


if __name__ == "__main__":
    main()
