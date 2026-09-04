"""Orchestration : retry, repli déterministe, fidélité de bout en bout."""

from __future__ import annotations

import unittest

from rag.models import ChunkPlan, PlannedChunk
from rag.planner import ChunkPlanner, HeuristicChunkPlanner, PlannerError
from rag.semantic_chunker import SemanticChunker
from rag.validator import PlanValidationError, PlanValidator
from tests.fixtures import DOCUMENT, parsed


class ScriptedPlanner(ChunkPlanner):
    """Planner de test : rejoue des plans (ou des exceptions) préparés."""

    def __init__(self, plans: list) -> None:
        self.plans = list(plans)
        self.feedbacks: list[str | None] = []
        self.appels = 0

    def plan(self, blocks, *, context_blocks=(), feedback=None) -> ChunkPlan:
        self.appels += 1
        self.feedbacks.append(feedback)
        resultat = self.plans.pop(0) if self.plans else ChunkPlan(chunks=[])
        if isinstance(resultat, Exception):
            raise resultat
        return resultat


def un_seul_chunk(document) -> ChunkPlan:
    ids = [b.block_id for b in document.indexable_blocks]
    return ChunkPlan(chunks=[PlannedChunk("C001", ids, [ids[0]])])


class TestSemanticChunker(unittest.TestCase):
    def setUp(self) -> None:
        self.document = parsed()

    def chunker(self, planner: ChunkPlanner, **kwargs) -> SemanticChunker:
        options = dict(max_chunk_chars=100_000, min_chunk_chars=0, max_attempts=2)
        options.update(kwargs)
        return SemanticChunker(planner=planner, **options)

    # --- cas nominal ---------------------------------------------------
    def test_document_correctement_chunke(self) -> None:
        planner = ScriptedPlanner([un_seul_chunk(self.document)])
        chunks = self.chunker(planner).chunk(self.document)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_id, "C001")
        self.assertEqual(chunk.source, "architecture.txt")
        self.assertEqual(chunk.section_path, ("Architecture de la plateforme",))
        self.assertEqual(chunk.block_ids, tuple(b.block_id for b in self.document.indexable_blocks))
        # Le texte est bien un assemblage littéral du document.
        for block in self.document.indexable_blocks:
            self.assertIn(block.text, chunk.text)
        self.assertNotIn("---", chunk.text)  # le séparateur n'est pas indexable
        self.assertEqual(chunk.line_start, 1)

    def test_couverture_totale_et_ordre(self) -> None:
        planner = HeuristicChunkPlanner(target_chunk_chars=150, max_chunk_chars=400)
        chunks = SemanticChunker(planner=planner, max_chunk_chars=400).chunk(self.document)

        vus = [bid for chunk in chunks for bid in chunk.block_ids]
        attendus = [b.block_id for b in self.document.indexable_blocks]
        self.assertEqual(vus, attendus)
        self.assertEqual(len(set(vus)), len(vus))

    def test_texte_reconstruit_depuis_le_document(self) -> None:
        chunks = SemanticChunker(planner=HeuristicChunkPlanner()).chunk(self.document)
        for chunk in chunks:
            for block_id in chunk.block_ids:
                block = self.document.by_id()[block_id]
                self.assertIn(block.text, chunk.text)
                self.assertEqual(
                    block.text, self.document.text[block.char_start:block.char_end]
                )

    # --- retry ----------------------------------------------------------
    def test_retry_avec_feedback_apres_plan_invalide(self) -> None:
        ids = [b.block_id for b in self.document.indexable_blocks]
        incomplet = ChunkPlan(chunks=[PlannedChunk("C001", ids[:-1], [])])  # bloc perdu
        planner = ScriptedPlanner([incomplet, un_seul_chunk(self.document)])

        chunks = self.chunker(planner).chunk(self.document)

        self.assertEqual(planner.appels, 2)
        self.assertIsNone(planner.feedbacks[0])
        self.assertIn("manquant", planner.feedbacks[1])
        self.assertEqual(len(chunks), 1)

    def test_retry_apres_erreur_du_planner(self) -> None:
        planner = ScriptedPlanner([PlannerError("JSON invalide"), un_seul_chunk(self.document)])
        chunks = self.chunker(planner).chunk(self.document)
        self.assertEqual(planner.appels, 2)
        self.assertIn("JSON invalide", planner.feedbacks[1])
        self.assertEqual(len(chunks), 1)

    # --- repli ----------------------------------------------------------
    def test_repli_deterministe_si_le_llm_echoue_toujours(self) -> None:
        halluciné = ChunkPlan(chunks=[PlannedChunk("C001", ["B404", "B405"], [])])
        planner = ScriptedPlanner([halluciné, halluciné, halluciné])

        chunks = self.chunker(planner, max_chunk_chars=400).chunk(self.document)

        self.assertEqual(planner.appels, 2)  # 2 tentatives, puis repli
        vus = [bid for chunk in chunks for bid in chunk.block_ids]
        self.assertEqual(vus, [b.block_id for b in self.document.indexable_blocks])

    def test_repli_si_le_llm_duplique_des_blocs(self) -> None:
        ids = [b.block_id for b in self.document.indexable_blocks]
        duplique = ChunkPlan(chunks=[PlannedChunk("C001", ids + [ids[0]], [])])
        planner = ScriptedPlanner([duplique, duplique])
        chunks = self.chunker(planner, max_chunk_chars=400).chunk(self.document)
        vus = [bid for chunk in chunks for bid in chunk.block_ids]
        self.assertEqual(vus, ids)

    def test_repli_si_le_llm_reordonne(self) -> None:
        ids = [b.block_id for b in self.document.indexable_blocks]
        desordre = ChunkPlan(chunks=[PlannedChunk("C001", [ids[1], ids[0]] + ids[2:], [])])
        planner = ScriptedPlanner([desordre, desordre])
        chunks = self.chunker(planner, max_chunk_chars=400).chunk(self.document)
        vus = [bid for chunk in chunks for bid in chunk.block_ids]
        self.assertEqual(vus, ids)

    # --- réparations déterministes --------------------------------------
    def test_chunk_trop_long_decoupe_sur_une_frontiere_de_bloc(self) -> None:
        planner = ScriptedPlanner([un_seul_chunk(self.document)])
        chunks = self.chunker(planner, max_chunk_chars=150).chunk(self.document)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            blocs = [self.document.by_id()[bid] for bid in chunk.block_ids]
            if len(blocs) > 1:
                self.assertLessEqual(sum(len(b.text) for b in blocs), 150)
        vus = [bid for chunk in chunks for bid in chunk.block_ids]
        self.assertEqual(vus, [b.block_id for b in self.document.indexable_blocks])
        self.assertEqual([c.chunk_id for c in chunks],
                         [f"C{i:03d}" for i in range(1, len(chunks) + 1)])

    def test_chunk_trop_court_fusionne(self) -> None:
        ids = [b.block_id for b in self.document.indexable_blocks]
        eclate = ChunkPlan(chunks=[PlannedChunk(f"C{i:03d}", [bid], []) for i, bid in enumerate(ids, 1)])
        planner = ScriptedPlanner([eclate])
        chunks = self.chunker(planner, min_chunk_chars=200, max_chunk_chars=2000).chunk(self.document)
        self.assertLess(len(chunks), len(ids))

    # --- découpage en lots -----------------------------------------------
    def test_document_long_decoupe_en_lots(self) -> None:
        texte = "Titre\n\n" + "\n\n".join(f"Paragraphe numero {i} du document." for i in range(30))
        document = parsed(texte, source="long.txt")
        planner = HeuristicChunkPlanner(target_chunk_chars=200, max_chunk_chars=400)
        chunker = SemanticChunker(planner=planner, max_chunk_chars=400, batch_blocks=8)

        chunks = chunker.chunk(document)
        vus = [bid for chunk in chunks for bid in chunk.block_ids]
        self.assertEqual(vus, [b.block_id for b in document.indexable_blocks])

    def test_document_sans_bloc_indexable(self) -> None:
        document = parsed("Titre\n\n---\n", source="vide.txt")
        document.blocks[:] = [b for b in document.blocks if not b.indexable]
        self.assertEqual(SemanticChunker(planner=HeuristicChunkPlanner()).chunk(document), [])

    # --- section_path -----------------------------------------------------
    def test_section_path_recale_sur_la_hierarchie_du_document(self) -> None:
        """Un chemin de section fantaisiste est remplacé par celui du document."""
        ids = [b.block_id for b in self.document.indexable_blocks]
        backend = next(b for b in self.document.blocks if b.text == "## Backend")
        suite = ids[ids.index(backend.block_id):]
        plan = ChunkPlan(
            chunks=[
                PlannedChunk("C001", ids[: ids.index(backend.block_id)], []),
                # Chemin inventé : un titre qui n'englobe pas ce chunk.
                PlannedChunk("C002", suite, ["B999"]),
            ]
        )
        chunks = self.chunker(ScriptedPlanner([plan])).chunk(self.document)

        self.assertEqual(chunks[1].section_path,
                         ("Architecture de la plateforme", "Backend"))
        # Aucune tentative supplémentaire : la correction est déterministe.
        self.assertEqual(len(chunks), 2)

    # --- garde-fou final ---------------------------------------------------
    def test_le_repli_lui_meme_est_valide(self) -> None:
        """Si même le repli produit un plan invalide, rien n'est indexé."""
        mauvais = ChunkPlan(chunks=[PlannedChunk("C001", ["B404"], [])])
        chunker = SemanticChunker(
            planner=ScriptedPlanner([mauvais, mauvais]),
            fallback_planner=ScriptedPlanner([mauvais]),
            max_attempts=2,
        )
        with self.assertRaises(PlanValidationError):
            chunker.chunk(self.document)


if __name__ == "__main__":
    unittest.main()
