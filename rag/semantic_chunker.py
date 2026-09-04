"""Orchestration du chunking sémantique.

    Document parsé
        -> lots de blocs
        -> plan du LLM            (planner.py)
        -> réparations déterministes (taille min/max)
        -> validation stricte     (validator.py)   --échec--> retry, puis repli
        -> reconstruction         (reconstructor.py)
        -> vérification finale du texte
        -> chunks

Aucune décision du LLM n'est appliquée sans être passée par le validateur, et
aucun texte ne provient du LLM. En dernier recours, un plan déterministe
(`HeuristicChunkPlanner`) garantit qu'un document est toujours indexable.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .models import Block, Chunk, ChunkPlan, ParsedDocument, PlannedChunk
from .planner import ChunkPlanner, HeuristicChunkPlanner, PlannerError
from .reconstructor import ChunkReconstructor
from .validator import PlanValidationError, PlanValidator

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Transforme un `ParsedDocument` en chunks fidèles et traçables."""

    def __init__(
        self,
        planner: ChunkPlanner,
        validator: PlanValidator | None = None,
        reconstructor: ChunkReconstructor | None = None,
        fallback_planner: ChunkPlanner | None = None,
        *,
        target_chunk_chars: int = 1200,
        max_chunk_chars: int = 2000,
        min_chunk_chars: int = 200,
        batch_blocks: int = 40,
        batch_chars: int = 12000,
        max_attempts: int = 2,
    ) -> None:
        self.planner = planner
        self.validator = validator or PlanValidator()
        self.reconstructor = reconstructor or ChunkReconstructor()
        self.fallback_planner = fallback_planner or HeuristicChunkPlanner(
            target_chunk_chars=target_chunk_chars, max_chunk_chars=max_chunk_chars
        )
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.batch_blocks = batch_blocks
        self.batch_chars = batch_chars
        self.max_attempts = max(1, max_attempts)

    # ------------------------------------------------------------------
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Découpe un document ; garantit une couverture de 100 % ou lève une erreur."""
        indexables = document.indexable_blocks
        if not indexables:
            return []

        planned: list[PlannedChunk] = []
        for batch in self._batches(indexables):
            context = [b for b in indexables if b.position < batch[0].position and b.is_heading]
            planned.extend(self._plan_batch(batch, context, document).chunks)

        plan = self._renumber(self._merge_small(planned, document))

        # Filet de sécurité : le plan complet est revalidé contre le document entier.
        self.validator.validate(plan, document.blocks, headings=document.blocks)

        chunks = self.reconstructor.build(plan, document)
        self.validator.verify_chunks(chunks, document, separator=self.reconstructor.separator)
        return chunks

    # ------------------------------------------------------------------
    # Planification d'un lot, avec retry puis repli déterministe
    # ------------------------------------------------------------------
    def _plan_batch(
        self,
        batch: list[Block],
        context: list[Block],
        document: ParsedDocument,
    ) -> ChunkPlan:
        feedback: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                plan = self.planner.plan(batch, context_blocks=context, feedback=feedback)
                plan = self._split_oversized(plan, document)
                plan = self._repair_section_paths(plan, document)
                self.validator.validate(plan, batch, headings=document.blocks)
                return plan
            except PlanValidationError as error:
                feedback = error.report()
                logger.warning(
                    "Plan rejeté (%s, tentative %d/%d) :\n%s",
                    document.source, attempt, self.max_attempts, feedback,
                )
            except PlannerError as error:
                feedback = str(error)
                logger.warning(
                    "Planner en échec (%s, tentative %d/%d) : %s",
                    document.source, attempt, self.max_attempts, error,
                )

        logger.warning("Repli sur le plan déterministe pour %s.", document.source)
        plan = self.fallback_planner.plan(batch, context_blocks=context)
        plan = self._split_oversized(plan, document)
        plan = self._repair_section_paths(plan, document)
        # Le repli est validé comme le reste : aucun chunk n'échappe au contrôle.
        self.validator.validate(plan, batch, headings=document.blocks)
        return plan

    # ------------------------------------------------------------------
    # Découpage en lots (contrainte de contexte du LLM)
    # ------------------------------------------------------------------
    def _batches(self, blocks: Sequence[Block]) -> list[list[Block]]:
        batches: list[list[Block]] = []
        current: list[Block] = []
        size = 0

        for block in blocks:
            too_many = len(current) >= self.batch_blocks
            too_long = size + len(block.text) > self.batch_chars
            # Un titre est une frontière naturelle : on y coupe si le lot est déjà bien rempli.
            natural = block.is_heading and len(current) >= self.batch_blocks // 2

            if current and (too_many or too_long or natural):
                batches.append(current)
                current, size = [], 0

            current.append(block)
            size += len(block.text)

        if current:
            batches.append(current)
        return batches

    # ------------------------------------------------------------------
    # Réparations déterministes (ne peuvent ni perdre ni réordonner un bloc)
    # ------------------------------------------------------------------
    def _split_oversized(self, plan: ChunkPlan, document: ParsedDocument) -> ChunkPlan:
        """Coupe un chunk trop long sur une frontière de bloc."""
        by_id = document.by_id()
        result: list[PlannedChunk] = []

        for planned in plan.chunks:
            current: list[str] = []
            size = 0
            part = 0

            def emit(block_ids: list[str]) -> None:
                nonlocal part
                # Les morceaux issus d'une coupure gardent un identifiant distinct
                # (C003, C003-2, ...) ; la numérotation finale est refaite ensuite.
                part += 1
                suffixe = "" if part == 1 else f"-{part}"
                result.append(
                    PlannedChunk(f"{planned.id}{suffixe}", block_ids, list(planned.section_path))
                )

            for block_id in planned.block_ids:
                block = by_id.get(block_id)
                length = len(block.text) if block else 0
                if current and size + length > self.max_chunk_chars:
                    emit(current)
                    current, size = [], 0
                current.append(block_id)
                size += length
            if current:
                emit(current)

        return ChunkPlan(chunks=result)

    def _repair_section_paths(self, plan: ChunkPlan, document: ParsedDocument) -> ChunkPlan:
        """Recale un `section_path` incohérent sur la hiérarchie réelle du document.

        Le chemin proposé par le planner est conservé s'il correspond exactement à
        la pile de titres réellement ouverts au début du chunk ; sinon il est
        remplacé par cette pile. La hiérarchie provient donc toujours du document —
        jamais d'une supposition du modèle — et une dérive de titres d'un lot à
        l'autre ne coûte pas un appel supplémentaire.
        """
        positions = {b.block_id: b.position for b in document.blocks}
        repaired: list[PlannedChunk] = []

        for chunk in plan.chunks:
            debut = min(
                (positions[bid] for bid in chunk.block_ids if bid in positions), default=0
            )
            canonique = self._heading_stack(document, debut)
            if list(chunk.section_path) != canonique:
                logger.debug(
                    "Chunk %s : section_path recalé sur la hiérarchie du document (%s -> %s).",
                    chunk.id, chunk.section_path, canonique,
                )
            repaired.append(PlannedChunk(chunk.id, list(chunk.block_ids), canonique))

        return ChunkPlan(chunks=repaired)

    @staticmethod
    def _heading_stack(document: ParsedDocument, position: int) -> list[str]:
        """Titres ouverts au bloc `position` (titre de tête du chunk inclus)."""
        stack: list[Block] = []
        for block in document.blocks:
            if block.position > position:
                break
            if block.is_heading:
                while stack and stack[-1].level >= block.level:
                    stack.pop()
                stack.append(block)
        return [block.block_id for block in stack]

    def _merge_small(
        self, planned: Sequence[PlannedChunk], document: ParsedDocument
    ) -> list[PlannedChunk]:
        """Fusionne un chunk trop court avec le suivant (ex. un titre resté seul)."""
        by_id = document.by_id()

        def size(chunk: PlannedChunk) -> int:
            return sum(len(by_id[bid].text) for bid in chunk.block_ids if bid in by_id)

        merged: list[PlannedChunk] = []
        for chunk in planned:
            fusionnable = (
                merged
                and size(merged[-1]) < self.min_chunk_chars
                and size(merged[-1]) + size(chunk) <= self.max_chunk_chars
                # On ne fusionne qu'à l'intérieur d'une même branche de titres
                # (typiquement : un titre resté seul et la section qu'il ouvre).
                and self._prefixe(merged[-1].section_path, chunk.section_path)
            )
            if fusionnable:
                previous = merged.pop()
                merged.append(
                    PlannedChunk(
                        id=previous.id,
                        block_ids=list(previous.block_ids) + list(chunk.block_ids),
                        # Le chunk commence là où commençait le précédent : on garde
                        # son chemin, seul garanti antérieur à son premier bloc.
                        section_path=list(previous.section_path),
                    )
                )
            else:
                merged.append(chunk)
        return merged

    @staticmethod
    def _prefixe(court: Sequence[str], long: Sequence[str]) -> bool:
        return list(court) == list(long[: len(court)])

    @staticmethod
    def _renumber(chunks: Sequence[PlannedChunk]) -> ChunkPlan:
        return ChunkPlan(
            chunks=[
                PlannedChunk(
                    id=f"C{position:03d}",
                    block_ids=list(chunk.block_ids),
                    section_path=list(chunk.section_path),
                )
                for position, chunk in enumerate(chunks, start=1)
            ]
        )
