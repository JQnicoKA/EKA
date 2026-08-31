"""Étape 1 du pipeline : le chargement des documents.

`DocumentLoader` est une interface abstraite : si demain on veut charger du
Markdown, du HTML ou du DOCX, il suffira d'ajouter une nouvelle classe qui
implémente `load()`. Le reste du pipeline n'aura pas à changer.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from pypdf import PdfReader

from .models import Page

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Contrat commun à tous les chargeurs de documents."""

    @abstractmethod
    def load(self, path: Path) -> list[Page]:
        """Retourne la liste des pages d'un document."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Indique si ce chargeur sait traiter ce fichier."""


class PDFLoader(DocumentLoader):
    """Chargeur de fichiers PDF basé sur `pypdf`.

    Le texte est extrait page par page : on conserve ainsi le numéro de page,
    indispensable pour citer la source exacte dans la réponse finale.
    """

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(self, path: Path) -> list[Page]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        if not self.supports(path):
            raise ValueError(f"Format non supporté par PDFLoader : {path.name}")

        reader = PdfReader(str(path))
        pages: list[Page] = []

        for page_number, pdf_page in enumerate(reader.pages, start=1):
            # `extract_text()` peut renvoyer None sur une page sans couche texte
            # (par exemple une page scannée : il faudrait un OCR, hors périmètre ici).
            raw_text = pdf_page.extract_text() or ""
            text = self._clean(raw_text)

            if not text:
                logger.warning("Page %d de %s : aucun texte extrait (page scannée ?)",
                               page_number, path.name)
                continue

            pages.append(Page(source=path.name, page=page_number, text=text))

        if not pages:
            raise ValueError(
                f"Aucun texte n'a pu être extrait de {path.name}. "
                "Le PDF est peut-être constitué d'images (OCR nécessaire)."
            )

        logger.info("%s : %d page(s) chargée(s)", path.name, len(pages))
        return pages

    @staticmethod
    def _clean(text: str) -> str:
        """Nettoyage minimal : césures de fin de ligne, espaces et sauts de ligne multiples."""
        # Recolle les mots coupés en fin de ligne ("informa-\ntion" -> "information").
        text = re.sub(r"-\n(?=\w)", "", text)
        # Uniformise les espaces horizontaux.
        text = re.sub(r"[ \t]+", " ", text)
        # Réduit les enchaînements de sauts de ligne à deux au maximum.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
