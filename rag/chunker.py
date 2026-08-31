"""Étape 2 du pipeline : le découpage en chunks.

Pourquoi découper ? Deux raisons :
  1. Un embedding représente d'autant moins bien un texte qu'il est long ;
     de petits morceaux donnent une recherche sémantique plus précise.
  2. On ne peut pas envoyer un document entier dans le prompt du LLM.

Stratégie retenue (volontairement simple) : découpage **page par page**, en
respectant autant que possible les frontières naturelles du texte (paragraphes,
puis phrases, puis mots). Chaque chunk appartient donc à une seule page, ce qui
permet de citer précisément la source.
"""

from __future__ import annotations

import logging
import re

from .models import Chunk, Page

logger = logging.getLogger(__name__)

# Séparateurs testés du plus "gros" au plus "fin". On coupe au niveau le plus
# élevé possible pour préserver le sens.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


class TextChunker:
    """Découpe des pages en chunks de taille homogène avec recouvrement."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, pages: list[Page]) -> list[Chunk]:
        """Transforme une liste de pages en une liste de chunks numérotés."""
        chunks: list[Chunk] = []
        index = 0

        for page in pages:
            for piece in self._split_text(page.text):
                chunks.append(
                    Chunk(source=page.source, page=page.page, index=index, text=piece)
                )
                index += 1

        logger.info("Découpage : %d page(s) -> %d chunk(s)", len(pages), len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Découpage d'un texte unique
    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> list[str]:
        """Découpe un texte en fenêtres de `chunk_size` avec `chunk_overlap`."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        pieces: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Si on n'est pas à la fin du texte, on recule jusqu'à une frontière propre.
            if end < len(text):
                end = self._find_break(text, start, end)

            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)

            if end >= len(text):
                break

            # Le chunk suivant démarre `chunk_overlap` caractères avant la fin
            # du précédent : ce recouvrement évite de couper une idée en deux.
            start = max(end - self.chunk_overlap, start + 1)

        return pieces

    def _find_break(self, text: str, start: int, end: int) -> int:
        """Cherche la meilleure position de coupure dans [start, end].

        On ne recule pas au-delà de la moitié du chunk : mieux vaut une coupure
        moins élégante qu'un chunk minuscule.
        """
        floor = start + self.chunk_size // 2

        for separator in _SEPARATORS:
            position = text.rfind(separator, floor, end)
            if position != -1:
                # On coupe *après* le séparateur pour ne pas le perdre.
                return position + len(separator)

        # Aucune frontière trouvée (mot très long, tableau...) : coupure brute.
        return end


def normalize_whitespace(text: str) -> str:
    """Utilitaire : compacte les espaces d'un texte (utile pour l'affichage)."""
    return re.sub(r"\s+", " ", text).strip()
