"""Étape 2 du pipeline : le découpage en chunks.

Pourquoi découper ? Deux raisons :
  1. Un embedding représente d'autant moins bien un texte qu'il est long ;
     de petits morceaux donnent une recherche sémantique plus précise.
  2. On ne peut pas envoyer un document entier dans le prompt du LLM.

Stratégie retenue (volontairement simple) : une fenêtre glissante de
`chunk_size` caractères avec un recouvrement de `chunk_overlap`, en coupant
autant que possible sur une frontière naturelle du texte (paragraphes, puis
phrases, puis mots).

Chaque chunk retient les numéros de ligne d'où il provient : c'est ce qui
permet d'afficher une référence précise (`notes.txt (l. 12-40)`).

  Les règles

  1. Le document est traité d'un bloc. Pas de découpage préalable par
  page ou par section — TextLoader renvoie le fichier entier, le chunker
  travaille sur cette chaîne unique.

  2. Fenêtre de 1 000 caractères maximum. On part de start, on vise
  start + 1000. C'est un plafond strict : aucun chunk ne dépasse
  chunk_size.

  3. On recule jusqu'à une frontière propre. Plutôt que couper à la
  position 1 000 exacte, on cherche le dernier séparateur avant elle,
  testé dans cet ordre de priorité :

  ┌───────┬────────────┬───────────────────┐
  │ Ordre │ Séparateur │     Intention     │
  ├───────┼────────────┼───────────────────┤
  │ 1     │ \n\n       │ fin de paragraphe │
  ├───────┼────────────┼───────────────────┤
  │ 2     │ \n         │ fin de ligne      │
  ├───────┼────────────┼───────────────────┤
  │ 3     │ .          │ fin de phrase     │
  ├───────┼────────────┼───────────────────┤
  │ 4     │ espace     │ fin de mot        │
  └───────┴────────────┴───────────────────┘

  Le premier trouvé gagne — donc on privilégie toujours la coupure la
  plus « haute » disponible. La coupe se fait après le séparateur, pour
  ne pas le perdre.

  4. On ne recule jamais au-delà de la moitié. Le plancher est start + 
  500. Si aucun séparateur n'existe entre 500 et 1 000, on coupe
  brutalement à 1 000. La raison : mieux vaut une coupure inélégante
  qu'un chunk minuscule. C'est pour ça que les chunks font en pratique
  entre 500 et 1 000 caractères.

  5. Recouvrement de 150 caractères. Le chunk suivant démarre à
  fin_du_précédent − 150. Les 150 derniers caractères d'un chunk sont
  donc répétés au début du suivant. C'est ce qui fait que le volume
  indexé dépasse de ~21 % le texte source — et c'est aussi ce qui
  explique les débuts de chunk en plein milieu d'un mot ('ikes near 
  7. Numéros de ligne. line_start et line_end sont déduits en comptant
  les \n avant l'offset. line_end regarde le dernier caractère du chunk
  (end − 1, la borne étant exclusive). C'est fiable parce que TextLoader
  ne nettoie pas le texte — toute normalisation d'espaces casserait
  cette correspondance.

  8. Garde-fous. chunk_overlap >= chunk_size lève une erreur au
  démarrage (ce serait une boucle infinie). Un texte vide ou blanc
  renvoie zéro chunk. Une progression minimale d'un caractère est forcée
  à chaque tour.

  Ce que les règles n'incluent pas

  Aucune notion de structure : titres, listes, tableaux, blocs de code
  sont traités comme du texte ordinaire. Une frontière de section n'a
  pas plus de valeur qu'une fin de ligne quelconque — c'est précisément
  la différence avec le découpage par blocs de la branche
  semantic-chunking.
"""

from __future__ import annotations

import logging
import re

from .models import Chunk, Document

logger = logging.getLogger(__name__)

# Séparateurs testés du plus "gros" au plus "fin". On coupe au niveau le plus
# élevé possible pour préserver le sens.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


class TextChunker:
    """Découpe un document en chunks de taille homogène avec recouvrement."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        """Transforme un document en une liste de chunks numérotés."""
        text = document.text
        chunks: list[Chunk] = []

        for index, (piece, start, end) in enumerate(self._split_text(text)):
            chunks.append(
                Chunk(
                    source=document.source,
                    index=index,
                    text=piece,
                    line_start=self._line_at(text, start),
                    # `end` est exclusif : on regarde le dernier caractère du chunk.
                    line_end=self._line_at(text, max(start, end - 1)),
                )
            )

        logger.info("Découpage de %s : %d chunk(s)", document.source, len(chunks))
        return chunks

    @staticmethod
    def _line_at(text: str, offset: int) -> int:
        """Numéro de ligne (indexé à 1) du caractère situé à `offset`."""
        return text.count("\n", 0, offset) + 1

    # ------------------------------------------------------------------
    # Découpage d'un texte unique
    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> list[tuple[str, int, int]]:
        """Découpe un texte en (morceau, offset_début, offset_fin).

        Les offsets pointent dans `text` non modifié : ils restent donc valides
        pour retrouver la ligne d'origine.
        """
        if not text.strip():
            return []

        pieces: list[tuple[str, int, int]] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Si on n'est pas à la fin du texte, on recule jusqu'à une frontière propre.
            if end < len(text):
                end = self._find_break(text, start, end)

            # Le morceau est nettoyé de ses espaces de bord, mais on conserve les
            # offsets réels correspondant au texte retenu.
            brut = text[start:end]
            piece = brut.strip()
            if piece:
                decalage = len(brut) - len(brut.lstrip())
                debut_reel = start + decalage
                pieces.append((piece, debut_reel, debut_reel + len(piece)))

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
