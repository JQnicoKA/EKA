"""Étape 1 du pipeline : le chargement des documents.

`DocumentLoader` est une interface abstraite : si demain on veut charger du
PDF, du HTML ou du DOCX, il suffira d'ajouter une nouvelle classe qui
implémente `load()`. Le reste du pipeline n'aura pas à changer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .models import Document

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Contrat commun à tous les chargeurs de documents."""

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Retourne le contenu d'un document."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Indique si ce chargeur sait traiter ce fichier."""


class TextLoader(DocumentLoader):
    """Chargeur de fichiers texte brut (`.txt`).

    Le contenu est renvoyé **tel quel**, sans nettoyage : c'est la condition
    pour que les numéros de ligne des chunks correspondent exactement à ceux
    du fichier ouvert dans un éditeur.
    """

    SUFFIXES = (".txt",)

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUFFIXES

    def load(self, path: Path) -> Document:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        if not self.supports(path):
            raise ValueError(
                f"Format non supporté : {path.name}. "
                f"Extensions acceptées : {', '.join(self.SUFFIXES)}."
            )

        # `errors="replace"` : un octet invalide isolé ne doit pas faire échouer
        # l'ingestion d'un corpus entier.
        raw = path.read_text(encoding=self.encoding, errors="replace")

        # Uniformise les fins de ligne Windows/Mac sans changer le nombre de lignes.
        text = raw.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise ValueError(f"Le fichier {path.name} est vide.")

        logger.info("%s : %d caractère(s), %d ligne(s)",
                    path.name, len(text), text.count("\n") + 1)
        return Document(source=path.name, text=text)
