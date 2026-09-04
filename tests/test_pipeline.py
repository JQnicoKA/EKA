"""Le pipeline d'ingestion : de bout en bout, sans API ni Qdrant."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.embedder import Embedder
from rag.generator import AnswerGenerator
from rag.models import Chunk, RetrievedChunk
from rag.parser import TextParser
from rag.pipeline import RAGPipeline
from rag.planner import HeuristicChunkPlanner
from rag.semantic_chunker import SemanticChunker
from rag.vector_store import VectorStore
from tests.fixtures import DOCUMENT


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self.appels: list[int] = []   # taille de chaque lot vectorisé

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts):
        self.appels.append(len(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


class MemoryStore(VectorStore):
    """Base vectorielle en mémoire : suffisante pour tester la plomberie."""

    def __init__(self) -> None:
        self.points: dict[str, Chunk] = {}

    def add(self, chunks, vectors) -> int:
        for chunk in chunks:
            self.points[f"{chunk.source}:{chunk.index}"] = chunk
        return len(chunks)

    def search(self, vector, top_k):
        return [RetrievedChunk(chunk=c, score=1.0) for c in list(self.points.values())[:top_k]]

    def delete_source(self, source: str) -> None:
        for key in [k for k, c in self.points.items() if c.source == source]:
            del self.points[key]

    def list_sources(self) -> list[str]:
        return sorted({chunk.source for chunk in self.points.values()})


class FakeGenerator(AnswerGenerator):
    def generate(self, question, contexts):
        return f"{len(contexts)} extrait(s)."


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore()
        self.pipeline = RAGPipeline(
            parser=TextParser(max_block_chars=600),
            chunker=SemanticChunker(
                planner=HeuristicChunkPlanner(target_chunk_chars=300, max_chunk_chars=600),
                max_chunk_chars=600,
            ),
            embedder=FakeEmbedder(),
            vector_store=self.store,
            generator=FakeGenerator(),
            top_k=2,
        )

    def ecrire(self, contenu: str, nom: str = "architecture.txt") -> Path:
        dossier = Path(tempfile.mkdtemp())
        chemin = dossier / nom
        chemin.write_text(contenu, encoding="utf-8")
        return chemin

    def test_ingestion_puis_question(self) -> None:
        nombre = self.pipeline.ingest_file(self.ecrire(DOCUMENT))
        self.assertGreater(nombre, 0)
        self.assertEqual(len(self.store.points), nombre)

        # Chaque morceau du texte stocké provient littéralement du document.
        for chunk in self.store.points.values():
            self.assertTrue(chunk.block_ids)
            for morceau in chunk.text.split("\n\n"):
                self.assertIn(morceau, DOCUMENT)

        reponse = self.pipeline.ask("Quel langage pour le backend ?")
        self.assertEqual(reponse.text, "2 extrait(s).")
        self.assertTrue(reponse.sources[0].chunk.reference.startswith("architecture.txt (l."))

    def test_reindexation_ne_laisse_pas_de_chunks_orphelins(self) -> None:
        chemin = self.ecrire(DOCUMENT)
        self.pipeline.ingest_file(chemin)

        chemin.write_text("Architecture de la plateforme\n\nDocument raccourci.\n", encoding="utf-8")
        nombre = self.pipeline.ingest_file(chemin)

        self.assertEqual(len(self.store.points), nombre)
        self.assertNotIn("FastAPI", " ".join(c.text for c in self.store.points.values()))

    def test_ingestion_dossier_groupe_les_embeddings(self) -> None:
        """Les chunks de plusieurs documents partent en un seul lot d'embeddings."""
        chemin = self.ecrire(DOCUMENT)
        for numero in range(2, 6):
            (chemin.parent / f"doc{numero}.txt").write_text(DOCUMENT, encoding="utf-8")

        embedder = self.pipeline.embedder
        resultats = self.pipeline.ingest_directory(chemin.parent, batch_chunks=256)

        self.assertEqual(len(resultats), 5)
        total = sum(resultats.values())
        self.assertEqual(len(self.store.points), total)
        # Un seul appel, et non un par document.
        self.assertEqual(embedder.appels, [total])

    def test_ingestion_dossier_vide_le_tampon_par_lots(self) -> None:
        chemin = self.ecrire(DOCUMENT)
        for numero in range(2, 5):
            (chemin.parent / f"doc{numero}.txt").write_text(DOCUMENT, encoding="utf-8")

        embedder = self.pipeline.embedder
        resultats = self.pipeline.ingest_directory(chemin.parent, batch_chunks=3)

        self.assertGreater(len(embedder.appels), 1)
        self.assertEqual(sum(embedder.appels), sum(resultats.values()))
        self.assertTrue(all(taille >= 3 for taille in embedder.appels[:-1]))

    def test_echec_dembedding_marque_les_documents_a_zero(self) -> None:
        chemin = self.ecrire(DOCUMENT)

        def echouer(textes):
            raise RuntimeError("429 rate limit")

        self.pipeline.embedder.embed_documents = echouer
        resultats = self.pipeline.ingest_directory(chemin.parent)

        self.assertEqual(resultats, {"architecture.txt": 0})
        self.assertEqual(self.store.points, {})

    def test_reprise_saute_les_documents_deja_indexes(self) -> None:
        chemin = self.ecrire(DOCUMENT)
        for numero in range(2, 5):
            (chemin.parent / f"doc{numero}.txt").write_text(DOCUMENT, encoding="utf-8")

        # Première passe : un seul document, comme une ingestion interrompue.
        self.pipeline.ingest_file(chemin)
        deja = len(self.store.points)
        self.pipeline.embedder.appels.clear()

        resultats = self.pipeline.ingest_directory(chemin.parent, resume=True)

        self.assertNotIn("architecture.txt", resultats)      # sauté
        self.assertEqual(len(resultats), 3)                  # les trois autres
        self.assertEqual(len(self.store.points), deja + sum(resultats.values()))

    def test_reprise_sur_base_vide_traite_tout(self) -> None:
        chemin = self.ecrire(DOCUMENT)
        resultats = self.pipeline.ingest_directory(chemin.parent, resume=True)
        self.assertEqual(list(resultats), ["architecture.txt"])

    def test_ingestion_dossier_ignore_les_autres_formats(self) -> None:
        chemin = self.ecrire(DOCUMENT)
        (chemin.parent / "image.png").write_bytes(b"\x89PNG")
        resultats = self.pipeline.ingest_directory(chemin.parent)
        self.assertEqual(list(resultats), ["architecture.txt"])


if __name__ == "__main__":
    unittest.main()
