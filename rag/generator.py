"""Étapes 6 et 7 du pipeline : construction du prompt et génération de la réponse.

Le principe du RAG tient en une phrase : on n'attend pas du LLM qu'il *sache*,
on lui fournit les passages pertinents et on lui demande de répondre
**uniquement** à partir de ceux-ci. Cela réduit fortement les hallucinations et
permet de citer les sources.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from mistralai import Mistral

from .models import RetrievedChunk
from .utils import retry

logger = logging.getLogger(__name__)


# Le prompt système fixe le rôle et les règles. C'est le principal levier de
# qualité d'un RAG simple : n'hésitez pas à l'ajuster à votre cas d'usage.
SYSTEM_PROMPT = """Tu es un assistant documentaire. Tu réponds à la question de \
l'utilisateur en te basant EXCLUSIVEMENT sur les extraits de documents fournis.

Règles impératives :
- N'utilise aucune connaissance extérieure aux extraits fournis.
- Si les extraits ne contiennent pas la réponse, dis-le clairement : \
"Les documents fournis ne permettent pas de répondre à cette question."
- Cite systématiquement les extraits utilisés avec leur numéro, au format [1], [2].
- Réponds en français, de façon concise et factuelle.
"""

USER_PROMPT_TEMPLATE = """Extraits de documents :
---------------------
{context}
---------------------

Question : {question}

Réponse (en citant les extraits utilisés) :"""


class AnswerGenerator(ABC):
    """Contrat commun à tous les générateurs de réponse."""

    @abstractmethod
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        """Produit une réponse textuelle à partir de la question et des extraits."""


class MistralGenerator(AnswerGenerator):
    """Génération via l'API de chat Mistral."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        temperature: float = 0.1,
    ) -> None:
        self._client = Mistral(api_key=api_key)
        self._model = model
        # Température basse : on veut des réponses fidèles aux sources,
        # pas de la créativité.
        self._temperature = temperature

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "Aucun document pertinent n'a été trouvé pour répondre à cette question."

        prompt = USER_PROMPT_TEMPLATE.format(
            context=self._format_context(contexts),
            question=question.strip(),
        )

        response = retry(
            lambda: self._client.chat.complete(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            ),
            description="l'appel au LLM Mistral",
        )

        content = response.choices[0].message.content
        logger.info("Réponse générée (%d caractères).", len(content or ""))
        return (content or "").strip()

    @staticmethod
    def _format_context(contexts: list[RetrievedChunk]) -> str:
        """Numérote les extraits pour que le LLM puisse les citer ([1], [2], ...)."""
        blocs = []
        for position, result in enumerate(contexts, start=1):
            chunk = result.chunk
            blocs.append(f"[{position}] Source : {chunk.reference}\n{chunk.text}")
        return "\n\n".join(blocs)
