"""Objets métier échangés entre les différents composants du RAG.

Ces dataclasses constituent le "contrat" entre les étages du pipeline :
loader -> chunker -> embedder -> vector store -> generator.
Elles évitent de faire circuler des dictionnaires anonymes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """Un document texte chargé depuis le disque.

    Contrairement à un PDF, un fichier texte n'a pas de pages : le repère
    utilisé pour citer un passage est le **numéro de ligne**.
    """

    source: str   # nom du fichier d'origine, ex. "notes_reunion.txt"
    text: str     # contenu intégral du fichier


@dataclass(frozen=True)
class Chunk:
    """Un morceau de texte prêt à être vectorisé et stocké."""

    source: str
    index: int        # position du chunk dans le document (0, 1, 2, ...)
    text: str
    line_start: int = 0   # première ligne du chunk dans le fichier (indexée à 1)
    line_end: int = 0     # dernière ligne du chunk

    @property
    def reference(self) -> str:
        """Référence lisible affichée à l'utilisateur, ex. "notes.txt (l. 12-40)"."""
        if not self.line_start:
            return self.source
        if self.line_end and self.line_end != self.line_start:
            return f"{self.source} (l. {self.line_start}-{self.line_end})"
        return f"{self.source} (l. {self.line_start})"


@dataclass(frozen=True)
class RetrievedChunk:
    """Un chunk retrouvé dans la base vectorielle, accompagné de son score."""

    chunk: Chunk
    score: float  # similarité cosinus renvoyée par Qdrant (1.0 = identique)


@dataclass(frozen=True)
class Answer:
    """Réponse finale renvoyée à l'utilisateur."""

    question: str
    text: str                                 # réponse générée par le LLM
    sources: list[RetrievedChunk] = field(default_factory=list)  # passages utilisés
