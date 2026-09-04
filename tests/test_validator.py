"""Le validateur doit rejeter toute sortie de planner infidèle au document."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rag.models import Chunk, ChunkPlan, PlannedChunk
from rag.reconstructor import ChunkReconstructor
from rag.validator import PlanValidationError, PlanValidator
from tests.fixtures import parsed


def plan_complet(document) -> ChunkPlan:
    """Un chunk par bloc indexable : couverture 100 %, ordre respecté."""
    return ChunkPlan(
        chunks=[
            PlannedChunk(id=f"C{i:03d}", block_ids=[b.block_id], section_path=[])
            for i, b in enumerate(document.indexable_blocks, start=1)
        ]
    )


class TestValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.document = parsed()
        self.validator = PlanValidator()
        self.blocks = self.document.blocks

    def valider(self, plan: ChunkPlan) -> None:
        self.validator.validate(plan, self.blocks, headings=self.blocks)

    def erreurs(self, plan: ChunkPlan) -> list[str]:
        with self.assertRaises(PlanValidationError) as capture:
            self.valider(plan)
        return capture.exception.errors

    # --- cas nominal ---------------------------------------------------
    def test_plan_valide_accepte(self) -> None:
        self.valider(plan_complet(self.document))

    # --- bloc manquant -------------------------------------------------
    def test_bloc_manquant(self) -> None:
        plan = plan_complet(self.document)
        ampute = ChunkPlan(chunks=plan.chunks[:-1])
        messages = " ".join(self.erreurs(ampute))
        self.assertIn("manquant", messages)
        self.assertIn("Couverture insuffisante", messages)

    # --- bloc dupliqué -------------------------------------------------
    def test_bloc_duplique(self) -> None:
        plan = plan_complet(self.document)
        double = ChunkPlan(chunks=list(plan.chunks) + [PlannedChunk("C999", ["B001"], [])])
        self.assertIn("dupliqué", " ".join(self.erreurs(double)))

    # --- mauvais ordre -------------------------------------------------
    def test_ordre_non_conserve(self) -> None:
        chunks = list(plan_complet(self.document).chunks)
        chunks[0], chunks[1] = chunks[1], chunks[0]
        self.assertIn("ordre", " ".join(self.erreurs(ChunkPlan(chunks))))

    def test_ordre_non_conserve_dans_un_chunk(self) -> None:
        ids = [b.block_id for b in self.document.indexable_blocks]
        plan = ChunkPlan(chunks=[PlannedChunk("C001", [ids[1], ids[0]] + ids[2:], [])])
        self.assertIn("réordonnés", " ".join(self.erreurs(plan)))

    # --- block_id inexistant -------------------------------------------
    def test_block_id_inexistant(self) -> None:
        plan = plan_complet(self.document)
        invente = ChunkPlan(chunks=list(plan.chunks) + [PlannedChunk("C999", ["B999"], [])])
        self.assertIn("inconnu", " ".join(self.erreurs(invente)))

    # --- bloc non indexable --------------------------------------------
    def test_bloc_non_indexable_refuse(self) -> None:
        separateur = next(b for b in self.blocks if not b.indexable)
        plan = plan_complet(self.document)
        pollue = ChunkPlan(chunks=list(plan.chunks) + [PlannedChunk("C999", [separateur.block_id], [])])
        self.assertIn("non indexable", " ".join(self.erreurs(pollue)))

    # --- section_path ---------------------------------------------------
    def test_section_path_doit_etre_un_titre_du_document(self) -> None:
        paragraphe = next(b for b in self.blocks if b.type == "paragraph")
        plan = ChunkPlan(
            chunks=[
                PlannedChunk(
                    id="C001",
                    block_ids=[b.block_id for b in self.document.indexable_blocks],
                    section_path=[paragraphe.block_id],
                )
            ]
        )
        self.assertIn("n'est pas un titre", " ".join(self.erreurs(plan)))

    def test_section_path_invente_refuse(self) -> None:
        plan = ChunkPlan(
            chunks=[
                PlannedChunk(
                    id="C001",
                    block_ids=[b.block_id for b in self.document.indexable_blocks],
                    section_path=["Architecture inventée"],
                )
            ]
        )
        self.assertIn("n'est pas un titre", " ".join(self.erreurs(plan)))

    def test_titre_posterieur_refuse(self) -> None:
        titres = [b for b in self.blocks if b.is_heading]
        premier = self.document.indexable_blocks[0]
        plan = ChunkPlan(chunks=[PlannedChunk("C001", [premier.block_id], [titres[-1].block_id])])
        # (plan incomplet : on ne regarde que le message lié au titre)
        self.assertIn("postérieur", " ".join(self.erreurs(plan)))

    def test_plan_vide_refuse(self) -> None:
        self.assertIn("aucun chunk", " ".join(self.erreurs(ChunkPlan(chunks=[]))))

    # --- règle 6 : texte reconstruit == texte source --------------------
    def test_texte_modifie_detecte(self) -> None:
        plan = plan_complet(self.document)
        chunks = ChunkReconstructor().build(plan, self.document)
        self.validator.verify_chunks(chunks, self.document)  # inchangé : accepté

        cible = next(i for i, c in enumerate(chunks) if "Python" in c.text)
        falsifie = list(chunks)
        falsifie[cible] = replace(chunks[cible], text=chunks[cible].text.replace("Python", "Java"))
        with self.assertRaises(PlanValidationError) as capture:
            self.validator.verify_chunks(falsifie, self.document)
        self.assertIn("différent des blocs sources", " ".join(capture.exception.errors))

    def test_texte_hallucine_detecte(self) -> None:
        invente = [
            Chunk(
                source=self.document.source,
                index=0,
                text="Le backend utilise Rust et une base Cassandra.",
                chunk_id="C001",
                block_ids=(self.document.indexable_blocks[0].block_id,),
            )
        ]
        with self.assertRaises(PlanValidationError):
            self.validator.verify_chunks(invente, self.document)


if __name__ == "__main__":
    unittest.main()
