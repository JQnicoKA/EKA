"""Le parser doit produire des blocs exactement superposables au document."""

from __future__ import annotations

import unittest

from rag.models import BlockType
from tests.fixtures import DOCUMENT, parsed


class TestParser(unittest.TestCase):
    def test_blocs_identifies_et_ordonnes(self) -> None:
        document = parsed()
        ids = [b.block_id for b in document.blocks]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids[0], "B001")
        self.assertEqual([b.position for b in document.blocks],
                         list(range(1, len(document.blocks) + 1)))

    def test_texte_de_chaque_bloc_provient_du_document(self) -> None:
        document = parsed()
        for block in document.blocks:
            self.assertEqual(
                document.text[block.char_start:block.char_end],
                block.text,
                f"offsets faux pour {block.block_id}",
            )

    def test_types_reconnus(self) -> None:
        document = parsed()
        types = {b.type for b in document.blocks}
        self.assertIn(BlockType.HEADING, types)
        self.assertIn(BlockType.PARAGRAPH, types)
        self.assertIn(BlockType.LIST, types)
        self.assertIn(BlockType.SEPARATOR, types)
        # La ligne "---" existe mais n'est pas indexable.
        self.assertTrue(all(b.indexable for b in document.indexable_blocks))

    def test_titres_et_niveaux(self) -> None:
        document = parsed()
        titres = [(b.text, b.level) for b in document.blocks if b.is_heading]
        self.assertEqual(titres[0], ("Architecture de la plateforme", 1))
        self.assertEqual(titres[1], ("## Backend", 2))

    def test_aucun_bloc_ne_depasse_la_taille_maximale(self) -> None:
        long_texte = "Titre\n\n" + " ".join(f"Phrase numero {i}." for i in range(400))
        document = parsed(long_texte, source="long.txt", max_block_chars=300)
        self.assertTrue(all(len(b.text) <= 300 for b in document.blocks))
        # Le contenu reste intégralement couvert, sans chevauchement.
        recompose = "".join(
            document.text[b.char_start:b.char_end] for b in document.blocks
        )
        self.assertEqual(
            "".join(recompose.split()), "".join(document.text.split())
        )

    def test_lignes_citees(self) -> None:
        document = parsed()
        premier = document.blocks[0]
        self.assertEqual(premier.line_start, 1)
        self.assertGreaterEqual(premier.line_end, premier.line_start)

    def test_document_vide_rejete(self) -> None:
        with self.assertRaises(ValueError):
            parsed("   \n\n  ", source="vide.txt")


if __name__ == "__main__":
    unittest.main()
