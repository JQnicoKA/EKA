"""Le planner LLM : sortie structurée, robustesse, refus du contenu textuel."""

from __future__ import annotations

import json
import unittest

from rag.planner import (
    CHUNK_PLAN_SCHEMA,
    HeuristicChunkPlanner,
    MistralChunkPlanner,
    OllamaChunkPlanner,
    PlannerError,
)
from rag.validator import PlanValidator
from tests.fixtures import parsed


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class FakeChat:
    """Faux client Mistral : rejoue des réponses préparées et garde les appels."""

    def __init__(self, reponses: list[str]) -> None:
        self.reponses = list(reponses)
        self.appels: list[dict] = []

    def complete(self, **kwargs):
        self.appels.append(kwargs)
        return _Response(self.reponses.pop(0))


class FakeClient:
    def __init__(self, reponses: list[str]) -> None:
        self.chat = FakeChat(reponses)


def json_plan(document) -> str:
    ids = [b.block_id for b in document.indexable_blocks]
    return json.dumps(
        {"chunks": [{"id": "C001", "block_ids": ids, "section_path": [ids[0]]}]}
    )


class TestMistralChunkPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.document = parsed()

    def planner(self, reponses: list[str]) -> MistralChunkPlanner:
        return MistralChunkPlanner(api_key="x", client=FakeClient(reponses))

    def test_sortie_structurée_demandee(self) -> None:
        planner = self.planner([json_plan(self.document)])
        planner.plan(self.document.indexable_blocks)
        appel = planner._client.chat.appels[0]
        self.assertEqual(appel["response_format"]["type"], "json_schema")
        self.assertEqual(appel["response_format"]["json_schema"]["schema"], CHUNK_PLAN_SCHEMA)
        self.assertTrue(appel["response_format"]["json_schema"]["strict"])
        self.assertEqual(appel["temperature"], 0.0)

    def test_le_prompt_interdit_de_generer_du_texte(self) -> None:
        planner = self.planner([json_plan(self.document)])
        planner.plan(self.document.indexable_blocks)
        systeme = planner._client.chat.appels[0]["messages"][0]["content"]
        for interdiction in ("paraphrase", "résume", "supprime", "duplique", "réordonne", "invente"):
            self.assertIn(interdiction, systeme.lower())

    def test_le_prompt_liste_les_block_id(self) -> None:
        planner = self.planner([json_plan(self.document)])
        planner.plan(self.document.indexable_blocks)
        utilisateur = planner._client.chat.appels[0]["messages"][1]["content"]
        for block in self.document.indexable_blocks:
            self.assertIn(f"[{block.block_id}]", utilisateur)

    def test_le_texte_renvoye_par_le_llm_est_ignore(self) -> None:
        """Même si le LLM ajoute un champ 'text', il n'est jamais lu."""
        ids = [b.block_id for b in self.document.indexable_blocks]
        reponse = json.dumps(
            {
                "chunks": [
                    {
                        "id": "C001",
                        "block_ids": ids,
                        "section_path": [],
                        "text": "Résumé inventé par le LLM.",
                    }
                ]
            }
        )
        plan = self.planner([reponse]).plan(self.document.indexable_blocks)
        self.assertFalse(hasattr(plan.chunks[0], "text"))
        self.assertEqual(plan.chunks[0].block_ids, ids)

    def test_reponse_entouree_de_balises_markdown(self) -> None:
        contenu = "```json\n" + json_plan(self.document) + "\n```"
        plan = self.planner([contenu]).plan(self.document.indexable_blocks)
        self.assertEqual(len(plan.chunks), 1)

    def test_json_invalide_leve_planner_error(self) -> None:
        with self.assertRaises(PlannerError):
            self.planner(["ceci n'est pas du JSON"]).plan(self.document.indexable_blocks)

    def test_schema_absent_leve_planner_error(self) -> None:
        with self.assertRaises(PlannerError):
            self.planner(['{"resultat": []}']).plan(self.document.indexable_blocks)

    def test_le_feedback_est_transmis_au_llm(self) -> None:
        planner = self.planner([json_plan(self.document)])
        planner.plan(self.document.indexable_blocks, feedback="- block_id manquant : B004.")
        utilisateur = planner._client.chat.appels[0]["messages"][1]["content"]
        self.assertIn("REJETÉE", utilisateur)
        self.assertIn("B004", utilisateur)


class FakeTransport:
    """Faux serveur Ollama : rejoue des réponses préparées et garde les payloads."""

    def __init__(self, reponses: list[str]) -> None:
        self.reponses = list(reponses)
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"message": {"content": self.reponses.pop(0)}}


class TestOllamaChunkPlanner(unittest.TestCase):
    """Le planner local suit exactement le même contrat que le planner Mistral."""

    def setUp(self) -> None:
        self.document = parsed()

    def planner(self, reponses: list[str], **kwargs) -> OllamaChunkPlanner:
        return OllamaChunkPlanner(transport=FakeTransport(reponses), **kwargs)

    def test_schema_impose_au_modele_local(self) -> None:
        planner = self.planner([json_plan(self.document)], model="llama3.2:3b", num_ctx=4096)
        planner.plan(self.document.indexable_blocks)

        payload = planner._transport.payloads[0]
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["format"], CHUNK_PLAN_SCHEMA)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"], {"temperature": 0.0, "num_ctx": 4096})

    def test_meme_prompt_que_le_planner_mistral(self) -> None:
        planner = self.planner([json_plan(self.document)])
        planner.plan(self.document.indexable_blocks)
        messages = planner._transport.payloads[0]["messages"]

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("paraphrase", messages[0]["content"].lower())
        for block in self.document.indexable_blocks:
            self.assertIn(f"[{block.block_id}]", messages[1]["content"])

    def test_le_texte_renvoye_par_le_modele_est_ignore(self) -> None:
        ids = [b.block_id for b in self.document.indexable_blocks]
        reponse = json.dumps(
            {"chunks": [{"id": "C001", "block_ids": ids, "section_path": [],
                         "text": "Contenu inventé localement."}]}
        )
        plan = self.planner([reponse]).plan(self.document.indexable_blocks)
        self.assertEqual(plan.chunks[0].block_ids, ids)
        self.assertFalse(hasattr(plan.chunks[0], "text"))

    def test_reponse_illisible_leve_planner_error(self) -> None:
        with self.assertRaises(PlannerError):
            self.planner(["je ne sais pas faire"]).plan(self.document.indexable_blocks)

    def test_serveur_injoignable_leve_planner_error(self) -> None:
        planner = OllamaChunkPlanner(url="http://localhost:1", timeout=1.0, attempts=1)
        with self.assertRaises(PlannerError):
            planner.plan(self.document.indexable_blocks)


class TestHeuristicChunkPlanner(unittest.TestCase):
    def test_plan_deterministe_valide(self) -> None:
        document = parsed()
        planner = HeuristicChunkPlanner(target_chunk_chars=200, max_chunk_chars=400)
        plan = planner.plan(document.indexable_blocks)
        PlanValidator().validate(plan, document.blocks, headings=document.blocks)
        self.assertEqual(plan, planner.plan(document.indexable_blocks))

    def test_les_titres_ouvrent_un_chunk(self) -> None:
        document = parsed()
        plan = HeuristicChunkPlanner().plan(document.indexable_blocks)
        titres = {b.block_id for b in document.blocks if b.is_heading}
        for chunk in plan.chunks:
            # Un titre n'apparaît jamais ailleurs qu'en tête de son chunk.
            for position, block_id in enumerate(chunk.block_ids):
                if block_id in titres:
                    self.assertEqual(position, 0)

    def test_section_path_hierarchique(self) -> None:
        document = parsed()
        plan = HeuristicChunkPlanner().plan(document.indexable_blocks)
        titre_racine = document.blocks[0].block_id
        for chunk in plan.chunks[1:]:
            self.assertEqual(chunk.section_path[0], titre_racine)


if __name__ == "__main__":
    unittest.main()
