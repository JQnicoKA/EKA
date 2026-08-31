"""Objets métier échangés entre les différents composants du RAG.

Ces dataclasses constituent le "contrat" entre les étages du pipeline :
loader -> chunker -> embedder -> vector store -> generator.
Elles évitent de faire circuler des dictionnaires anonymes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Une page brute extraite d'un document PDF."""

    source: str   # nom du fichier d'origine, ex. "rapport_2024.pdf"
    page: int     # numéro de page, indexé à partir de 1
    text: str     # texte brut de la page


@dataclass(frozen=True)
class Chunk:
    """Un morceau de texte prêt à être vectorisé et stocké."""

    source: str
    page: int
    index: int    # position du chunk dans le document (0, 1, 2, ...)
    text: str

    @property
    def reference(self) -> str:
        """Référence lisible affichée à l'utilisateur, ex. "rapport.pdf (p. 3)"."""
        return f"{self.source} (p. {self.page})"


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
