"""Étape 3 : VALIDATION STRICTE du plan produit par le planner.

Ce module ne connaît ni le LLM ni le prompt : il compare un `ChunkPlan` aux
blocs réellement extraits du document. C'est le garde-fou anti-hallucination du
pipeline. Un plan invalide n'atteint jamais les embeddings.

Sept règles, dans l'ordre demandé :
  1. tous les blocs indexables du document sont présents ;
  2. aucun bloc n'est perdu ;
  3. aucun bloc n'est dupliqué ;
  4. l'ordre des blocs est conservé ;
  5. tous les block_id renvoyés existent dans le document ;
  6. le texte reconstruit correspond exactement au texte source
     (`verify_chunks`, après reconstruction) ;
  7. la couverture des blocs indexables est de 100 %.

À quoi s'ajoutent des règles de forme : identifiants de chunks uniques, chunks
non vides, `section_path` composé de titres réels du document et antérieurs au
chunk.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from .models import Block, Chunk, ChunkPlan, ParsedDocument

logger = logging.getLogger(__name__)


class PlanValidationError(ValueError):
    """Le plan viole au moins une règle de fidélité au document."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__(" | ".join(self.errors))

    def report(self) -> str:
        """Message multi-lignes destiné au prompt de réparation du LLM."""
        return "\n".join(f"- {error}" for error in self.errors)


class PlanValidator:
    """Vérifie qu'un plan couvre fidèlement les blocs d'un document."""

    def __init__(self, max_errors_reported: int = 10) -> None:
        self._max_errors = max_errors_reported

    # ------------------------------------------------------------------
    # Validation du plan (avant reconstruction)
    # ------------------------------------------------------------------
    def validate(
        self,
        plan: ChunkPlan,
        blocks: Sequence[Block],
        *,
        headings: Iterable[Block] | None = None,
    ) -> None:
        """Lève `PlanValidationError` si le plan n'est pas exactement fidèle.

        `blocks` : les blocs que le plan devait couvrir (document ou lot).
        `headings` : titres utilisables dans un `section_path` ; par défaut les
        titres de `blocks` (à élargir au document entier en mode par lots).
        """
        errors: list[str] = []
        expected = [b.block_id for b in blocks if b.indexable]
        known = {b.block_id: b for b in blocks}
        heading_blocks = {
            b.block_id: b for b in (headings if headings is not None else blocks) if b.is_heading
        }

        # --- forme générale ---
        if not plan.chunks:
            errors.append("Le plan ne contient aucun chunk.")

        seen_ids: set[str] = set()
        for position, chunk in enumerate(plan.chunks, start=1):
            if not chunk.block_ids:
                errors.append(f"Chunk {chunk.id or position} : aucun block_id.")
            if not chunk.id:
                errors.append(f"Chunk n°{position} : identifiant manquant.")
            elif chunk.id in seen_ids:
                errors.append(f"Identifiant de chunk dupliqué : {chunk.id}.")
            seen_ids.add(chunk.id)

        produced = plan.all_block_ids

        # --- règle 5 : les block_id existent ---
        unknown = [bid for bid in produced if bid not in known]
        for bid in unknown[: self._max_errors]:
            errors.append(f"block_id inconnu (inventé ?) : {bid}.")

        # --- blocs non indexables : ils n'ont rien à faire dans un chunk ---
        not_indexable = [bid for bid in produced if bid in known and not known[bid].indexable]
        for bid in not_indexable[: self._max_errors]:
            errors.append(f"block_id non indexable inclus dans un chunk : {bid}.")

        # --- règle 3 : pas de doublon ---
        counts: dict[str, int] = {}
        for bid in produced:
            counts[bid] = counts.get(bid, 0) + 1
        duplicates = sorted(bid for bid, count in counts.items() if count > 1)
        for bid in duplicates[: self._max_errors]:
            errors.append(f"block_id dupliqué : {bid} (présent {counts[bid]} fois).")

        # --- règles 1, 2 et 7 : couverture complète ---
        missing = [bid for bid in expected if bid not in counts]
        for bid in missing[: self._max_errors]:
            errors.append(f"block_id manquant (bloc perdu) : {bid}.")
        if expected:
            covered = len({bid for bid in produced if bid in set(expected)})
            coverage = covered / len(expected)
            if coverage < 1.0:
                errors.append(
                    f"Couverture insuffisante : {coverage:.1%} des blocs indexables "
                    f"({covered}/{len(expected)})."
                )

        # --- règle 4 : ordre conservé ---
        if not errors and produced != expected:
            errors.append(
                "L'ordre des blocs n'est pas conservé : "
                f"attendu {self._preview(expected)}, reçu {self._preview(produced)}."
            )

        # --- chunks contigus (conséquence de l'ordre, vérifiée explicitement) ---
        positions = {b.block_id: b.position for b in blocks}
        for chunk in plan.chunks:
            ranks = [positions[bid] for bid in chunk.block_ids if bid in positions]
            if len(ranks) > 1 and ranks != sorted(ranks):
                errors.append(f"Chunk {chunk.id} : blocs réordonnés à l'intérieur du chunk.")

        # --- section_path : uniquement des titres réels, antérieurs au chunk ---
        for chunk in plan.chunks:
            ranks_du_chunk = [positions[bid] for bid in chunk.block_ids if bid in positions]
            first_rank = min(ranks_du_chunk, default=None)
            last_rank = max(ranks_du_chunk, default=None)
            previous_rank = -1
            for bid in chunk.section_path:
                heading = heading_blocks.get(bid)
                if heading is None:
                    errors.append(
                        f"Chunk {chunk.id} : section_path contient '{bid}', "
                        "qui n'est pas un titre du document."
                    )
                    continue
                # Un titre du chemin précède le chunk, ou bien lui appartient
                # (cas du titre placé en tête du chunk).
                interne = bid in chunk.block_ids
                if last_rank is not None and heading.position > last_rank:
                    errors.append(
                        f"Chunk {chunk.id} : le titre {bid} est postérieur au contenu du chunk."
                    )
                elif first_rank is not None and heading.position > first_rank and not interne:
                    errors.append(
                        f"Chunk {chunk.id} : le titre {bid} n'englobe pas le début du chunk."
                    )
                if heading.position <= previous_rank:
                    errors.append(f"Chunk {chunk.id} : section_path désordonné autour de {bid}.")
                previous_rank = heading.position

        if errors:
            raise PlanValidationError(errors[: self._max_errors + 5])

    # ------------------------------------------------------------------
    # Vérification finale (après reconstruction) — règle 6
    # ------------------------------------------------------------------
    def verify_chunks(
        self,
        chunks: Sequence[Chunk],
        document: ParsedDocument,
        *,
        separator: str = "\n\n",
    ) -> None:
        """Vérifie que chaque chunk est bien un assemblage littéral du document."""
        errors: list[str] = []
        by_id = document.by_id()

        for chunk in chunks:
            blocks = [by_id.get(bid) for bid in chunk.block_ids]
            if any(block is None for block in blocks):
                errors.append(f"Chunk {chunk.chunk_id} : bloc source introuvable.")
                continue

            for block in blocks:
                if document.text[block.char_start:block.char_end] != block.text:
                    errors.append(
                        f"Chunk {chunk.chunk_id} : le bloc {block.block_id} ne correspond "
                        "plus au texte du document."
                    )

            expected = separator.join(block.text for block in blocks)
            if chunk.text != expected:
                errors.append(
                    f"Chunk {chunk.chunk_id} : texte reconstruit différent des blocs sources."
                )
            if not chunk.text.strip():
                errors.append(f"Chunk {chunk.chunk_id} : texte vide.")

        if errors:
            raise PlanValidationError(errors[: self._max_errors])

    # ------------------------------------------------------------------
    @staticmethod
    def _preview(ids: Sequence[str], limit: int = 8) -> str:
        head = ", ".join(ids[:limit])
        return head + (", ..." if len(ids) > limit else "")
