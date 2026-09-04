"""Étape 1 : SOURCE -> BLOCS.

Le parser transforme un fichier texte en une liste de `Block` identifiés
(`B001`, `B002`, ...). C'est la seule étape qui touche au fichier d'origine ;
tout le reste du pipeline travaille sur ces blocs.

Invariants garantis ici (le validateur les revérifie plus tard) :
  - `block.text == document.text[block.char_start:block.char_end]` ;
  - les blocs sont ordonnés et ne se chevauchent pas ;
  - aucun bloc ne dépasse `max_block_chars` (les longs paragraphes sont
    découpés sur des frontières de phrases) : un bloc est donc toujours
    assez petit pour tenir dans un chunk.

Le format visé est le `.txt` (éventuellement `.md`), tel que produit par les
exports d'outils d'entreprise : markdown léger, listes, blocs de code,
en-têtes `Clé: valeur`, transcriptions `Prénom: message`.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .models import Block, BlockType, ParsedDocument

logger = logging.getLogger(__name__)

# --- Motifs de reconnaissance -----------------------------------------
_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_SEPARATOR = re.compile(r"^([-*_=])\1{2,}$")
_BULLET = re.compile(r"^([-*•+]|\d+[.)]|[a-z][.)])\s+\S")
_TABLE_ROW = re.compile(r"^\|.*\|$|\S+(\s{2,}\S+){2,}$")
_FENCE = re.compile(r"^(```|~~~)")
# "Attendees", "Issue summary:", "Steps to reproduce (observed pattern):"
_TITLE_LINE = re.compile(r"^[A-Z0-9][^.!?]{0,79}:?$")
# Fin de phrase : point/!/? suivi d'un espace et d'une majuscule, ou fin de texte.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9«\"])")

_MAX_TITLE_CHARS = 80
_MAX_DOCUMENT_TITLE_CHARS = 200  # la première ligne d'un document est souvent longue


class DocumentParser(ABC):
    """Contrat commun à tous les parsers de documents."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Découpe un fichier en blocs structurés."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Indique si ce parser sait traiter ce fichier."""


class _Line:
    """Une ligne non vide, avec ses offsets exacts dans le texte du document."""

    __slots__ = ("text", "start", "end", "number")

    def __init__(self, text: str, start: int, end: int, number: int) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.number = number


class TextParser(DocumentParser):
    """Parser de fichiers texte (`.txt`, `.md`, `.markdown`).

    Le découpage en blocs suit la mise en forme du document : une ligne de
    titre, un paragraphe, une liste contiguë, un bloc de code délimité par des
    ``` ou un tableau forment chacun un bloc.
    """

    SUFFIXES = (".txt", ".md", ".markdown", ".text")

    def __init__(self, max_block_chars: int = 1200, encoding: str = "utf-8") -> None:
        if max_block_chars < 200:
            raise ValueError("max_block_chars doit valoir au moins 200 caractères.")
        self.max_block_chars = max_block_chars
        self.encoding = encoding

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUFFIXES

    # ------------------------------------------------------------------
    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        if not self.supports(path):
            raise ValueError(
                f"Format non supporté par TextParser : {path.name} "
                f"(extensions acceptées : {', '.join(self.SUFFIXES)})"
            )

        raw = path.read_text(encoding=self.encoding, errors="replace")
        return self.parse_text(raw, source=path.name)

    def parse_text(self, raw: str, source: str) -> ParsedDocument:
        """Variante utilisable sans fichier (tests, contenus déjà en mémoire)."""
        text = self._normalize(raw)
        if not text.strip():
            raise ValueError(f"Aucun texte exploitable dans {source}.")

        blocks: list[Block] = []
        for span in self._spans(text):
            blocks.extend(self._to_blocks(span, text, source, next_position=len(blocks) + 1))

        if not blocks:
            raise ValueError(f"Aucun bloc exploitable dans {source}.")

        document = ParsedDocument(source=source, text=text, blocks=blocks)
        self._audit(document)
        logger.info("%s : %d bloc(s) extrait(s).", source, len(blocks))
        return document

    # ------------------------------------------------------------------
    # Normalisation : on fige le texte de référence une fois pour toutes.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace(" ", " ")
        text = re.sub(r"[ \t]+(?=\n)", "", text)   # espaces en fin de ligne
        text = re.sub(r"\n{3,}", "\n\n", text)     # au plus une ligne vide
        return text.strip()

    # ------------------------------------------------------------------
    # Regroupement des lignes en "spans" (type + première/dernière ligne)
    # ------------------------------------------------------------------
    def _spans(self, text: str) -> list[tuple[str, int, list[_Line]]]:
        """Retourne des tuples (type, niveau, lignes) dans l'ordre du document."""
        lines = self._lines(text)
        spans: list[tuple[str, int, list[_Line]]] = []

        index = 0
        first_block = True
        last_heading_level = 0

        while index < len(lines):
            line = lines[index]

            # --- bloc de code délimité par ``` ou ~~~ ---
            if _FENCE.match(line.text):
                group = [line]
                index += 1
                while index < len(lines):
                    group.append(lines[index])
                    closed = _FENCE.match(lines[index].text)
                    index += 1
                    if closed:
                        break
                spans.append((BlockType.CODE, 0, group))
                first_block = False
                continue

            # --- ligne de séparation ---
            if _SEPARATOR.match(line.text):
                spans.append((BlockType.SEPARATOR, 0, [line]))
                index += 1
                continue

            # --- titre markdown ---
            markdown = _MD_HEADING.match(line.text)
            if markdown:
                last_heading_level = len(markdown.group(1))
                spans.append((BlockType.HEADING, last_heading_level, [line]))
                index += 1
                first_block = False
                continue

            # --- titre implicite (première ligne du document, ligne courte) ---
            level = self._implicit_heading_level(
                lines, index, first_block=first_block, last_heading_level=last_heading_level
            )
            if level:
                last_heading_level = max(last_heading_level, level)
                spans.append((BlockType.HEADING, level, [line]))
                index += 1
                first_block = False
                continue

            # --- liste : lignes consécutives commençant par une puce ---
            if _BULLET.match(line.text):
                group = [line]
                index += 1
                while index < len(lines) and self._continues_list(lines, index):
                    group.append(lines[index])
                    index += 1
                spans.append((BlockType.LIST, 0, group))
                first_block = False
                continue

            # --- tableau : lignes consécutives en colonnes ---
            if _TABLE_ROW.match(line.text):
                group = [line]
                index += 1
                while index < len(lines) and _TABLE_ROW.match(lines[index].text) \
                        and self._contiguous(lines[index - 1], lines[index]):
                    group.append(lines[index])
                    index += 1
                spans.append((BlockType.TABLE, 0, group))
                first_block = False
                continue

            # --- paragraphe : lignes consécutives jusqu'à la prochaine rupture ---
            group = [line]
            index += 1
            while index < len(lines) and self._continues_paragraph(lines, index):
                group.append(lines[index])
                index += 1
            spans.append((BlockType.PARAGRAPH, 0, group))
            first_block = False

        return spans

    @staticmethod
    def _lines(text: str) -> list[_Line]:
        """Lignes non vides, avec offsets absolus (espaces de bord exclus)."""
        result: list[_Line] = []
        offset = 0
        for number, raw_line in enumerate(text.split("\n"), start=1):
            stripped = raw_line.strip()
            if stripped:
                start = offset + (len(raw_line) - len(raw_line.lstrip()))
                result.append(_Line(stripped, start, start + len(stripped), number))
            offset += len(raw_line) + 1  # +1 pour le "\n"
        return result

    @staticmethod
    def _contiguous(previous: _Line, current: _Line) -> bool:
        """Vrai si les deux lignes se suivent sans ligne vide entre elles."""
        return current.number == previous.number + 1

    def _continues_list(self, lines: list[_Line], index: int) -> bool:
        current, previous = lines[index], lines[index - 1]
        if not self._contiguous(previous, current):
            return False
        if _MD_HEADING.match(current.text) or _SEPARATOR.match(current.text):
            return False
        # Nouvelle puce, ou continuation indentée de la puce précédente.
        return bool(_BULLET.match(current.text)) or current.start > previous.start

    def _continues_paragraph(self, lines: list[_Line], index: int) -> bool:
        current = lines[index]
        if not self._contiguous(lines[index - 1], current):
            return False
        return not (
            _MD_HEADING.match(current.text)
            or _SEPARATOR.match(current.text)
            or _BULLET.match(current.text)
            or _FENCE.match(current.text)
            or self._is_speaker_turn(current.text)
        )

    @staticmethod
    def _is_speaker_turn(text: str) -> bool:
        """Détecte "Alex: message" / "From: ..." : chaque tour de parole est un bloc."""
        return bool(re.match(r"^[A-Z][\w .'\-]{0,30}:\s+\S", text))

    def _implicit_heading_level(
        self, lines: list[_Line], index: int, *, first_block: bool, last_heading_level: int
    ) -> int:
        """Niveau d'un titre non markdown, ou 0 si la ligne n'en est pas un.

        Une fausse détection dégrade seulement la qualité du `section_path` :
        elle ne peut ni perdre ni altérer du texte.
        """
        line = lines[index]
        if first_block:
            # La première ligne d'un fichier texte est presque toujours son titre.
            return 1 if len(line.text) <= _MAX_DOCUMENT_TITLE_CHARS else 0
        if len(line.text) > _MAX_TITLE_CHARS:
            return 0
        if not _TITLE_LINE.match(line.text):
            return 0
        if _BULLET.match(line.text) or self._is_speaker_turn(line.text):
            return 0
        # Un titre est isolé de ce qui précède et suivi de contenu.
        if index > 0 and self._contiguous(lines[index - 1], line):
            return 0
        if index + 1 >= len(lines):
            return 0
        if not line.text.endswith(":") and not line.text.istitle() and not line.text.isupper():
            return 0
        return max(2, last_heading_level + 1)

    # ------------------------------------------------------------------
    # Conversion d'un span en un ou plusieurs blocs
    # ------------------------------------------------------------------
    def _to_blocks(
        self,
        span: tuple[str, int, list[_Line]],
        text: str,
        source: str,
        next_position: int,
    ) -> list[Block]:
        block_type, level, lines = span
        start, end = lines[0].start, lines[-1].end

        pieces = [(start, end)]
        if end - start > self.max_block_chars and block_type != BlockType.HEADING:
            pieces = self._split_span(text, lines, block_type)

        blocks: list[Block] = []
        for offset, (piece_start, piece_end) in enumerate(pieces):
            blocks.append(
                Block(
                    block_id=f"B{next_position + offset:03d}",
                    type=block_type,
                    text=text[piece_start:piece_end],
                    position=next_position + offset,
                    source=source,
                    char_start=piece_start,
                    char_end=piece_end,
                    line_start=self._line_number(lines, piece_start, first=True),
                    line_end=self._line_number(lines, piece_end, first=False),
                    level=level,
                )
            )
        return blocks

    def _split_span(
        self, text: str, lines: list[_Line], block_type: str
    ) -> list[tuple[int, int]]:
        """Découpe un span trop long en morceaux, sur des frontières naturelles.

        Priorité : frontière de ligne (listes, tableaux, code), puis frontière de
        phrase, puis coupure brute. Les morceaux restent contigus et exacts.
        """
        if block_type in (BlockType.LIST, BlockType.TABLE, BlockType.CODE) and len(lines) > 1:
            candidates = [line.start for line in lines[1:]]
        else:
            span_start, span_end = lines[0].start, lines[-1].end
            candidates = [
                span_start + match.start()
                for match in _SENTENCE_END.finditer(text[span_start:span_end])
            ]

        return self._pack(text, lines[0].start, lines[-1].end, candidates)

    def _pack(
        self, text: str, start: int, end: int, candidates: list[int]
    ) -> list[tuple[int, int]]:
        """Regroupe [start, end) en morceaux <= max_block_chars aux offsets fournis."""
        pieces: list[tuple[int, int]] = []
        cursor = start

        while end - cursor > self.max_block_chars:
            limit = cursor + self.max_block_chars
            cut = max((c for c in candidates if cursor < c <= limit), default=0)
            if not cut:
                # Aucune frontière naturelle : on coupe au dernier espace, sinon net.
                space = text.rfind(" ", cursor, limit) + 1
                cut = space if space > cursor else limit
            pieces.append((cursor, cut))
            cursor = cut

        pieces.append((cursor, end))
        return [(a, b) for a, b in pieces if b > a]

    @staticmethod
    def _line_number(lines: list[_Line], offset: int, *, first: bool) -> int:
        """Numéro de ligne correspondant à un offset, borné au span."""
        if first:
            for line in lines:
                if offset <= line.end:
                    return line.number
            return lines[-1].number
        for line in reversed(lines):
            if offset >= line.start:
                return line.number
        return lines[0].number

    # ------------------------------------------------------------------
    # Audit interne : le parser ne se fait pas confiance
    # ------------------------------------------------------------------
    @staticmethod
    def _audit(document: ParsedDocument) -> None:
        previous_end = -1
        for position, block in enumerate(document.blocks, start=1):
            if block.position != position:
                raise RuntimeError(f"Bloc {block.block_id} : position incohérente.")
            if block.char_start < previous_end:
                raise RuntimeError(f"Bloc {block.block_id} : chevauchement détecté.")
            if document.text[block.char_start:block.char_end] != block.text:
                raise RuntimeError(
                    f"Bloc {block.block_id} : le texte ne correspond pas au document source."
                )
            previous_end = block.char_end


def default_parser(max_block_chars: int = 1200) -> DocumentParser:
    """Parser utilisé par défaut dans le pipeline."""
    return TextParser(max_block_chars=max_block_chars)
