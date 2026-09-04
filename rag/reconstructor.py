"""Étape 4 : RECONSTRUCTION DÉTERMINISTE des chunks.

Le texte d'un chunk est **exclusivement** l'assemblage des blocs sources, dans
l'ordre du document. Aucun champ produit par le LLM n'est utilisé ici, hormis la
liste des `block_id` et le `section_path` (lui-même composé d'identifiants de
titres, résolus en texte depuis le document).

    C002 -> ["B003", "B004"] -> source[B003].text + source[B004].text -> chunk.text
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from .models import Chunk, ChunkPlan, ParsedDocument

logger = logging.getLogger(__name__)

_HEADING_DECORATION = re.compile(r"^#{1,6}\s*|\s*:$")


class ChunkReconstructor:
    """Transforme un plan validé en chunks prêts pour l'embedding."""

    def __init__(self, separator: str = "\n\n") -> None:
        self.separator = separator

    def build(
        self,
        plan: ChunkPlan,
        document: ParsedDocument,
        *,
        start_index: int = 0,
    ) -> list[Chunk]:
        by_id = document.by_id()
        chunks: list[Chunk] = []

        for offset, planned in enumerate(plan.chunks):
            blocks = [by_id[bid] for bid in planned.block_ids]  # KeyError impossible après validation
            index = start_index + offset

            chunks.append(
                Chunk(
                    source=document.source,
                    index=index,
                    # Le texte vient du document, jamais du LLM.
                    text=self.separator.join(block.text for block in blocks),
                    chunk_id=f"C{index + 1:03d}",
                    line_start=min(block.line_start for block in blocks),
                    line_end=max(block.line_end for block in blocks),
                    section_path=self._resolve_section_path(planned.section_path, by_id),
                    block_ids=tuple(planned.block_ids),
                )
            )

        logger.info("Reconstruction : %d chunk(s) depuis %s.", len(chunks), document.source)
        return chunks

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_section_path(block_ids: Sequence[str], by_id: dict) -> tuple[str, ...]:
        """Traduit des block_id de titres en textes de titres issus du document."""
        titres: list[str] = []
        for block_id in block_ids:
            block = by_id.get(block_id)
            if block is None or not block.is_heading:
                continue  # écarté par le validateur en amont ; ceinture et bretelles
            titre = _HEADING_DECORATION.sub("", " ".join(block.text.split())).strip()
            if titre:
                titres.append(titre)
        return tuple(titres)
