"""Étape 2 : BLOCS -> PLAN DE DÉCOUPAGE.

Le planner ne produit **aucun texte** : il retourne uniquement des décisions de
regroupement (`quels blocs vont ensemble`, `sous quels titres`). Le texte des
chunks est reconstruit ensuite, par du code déterministe, depuis le document
original (voir `reconstructor.py`).

Trois implémentations :
  - `MistralChunkPlanner` : API Mistral, sortie JSON contrainte par un schéma strict ;
  - `OllamaChunkPlanner` : modèle local servi par Ollama, même schéma imposé — aucun
    crédit consommé, le découpage tourne sur la machine ;
  - `HeuristicChunkPlanner` : plan déterministe fondé sur les titres et la taille,
    utilisé comme repli quand le LLM échoue (ou en mode « sans LLM »).

Toutes produisent le même objet `ChunkPlan`, passent par le même validateur et la
même reconstruction : aucun modèle n'a de privilège.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Sequence

from .models import Block, ChunkPlan, PlannedChunk
from .utils import retry

logger = logging.getLogger(__name__)


class PlannerError(RuntimeError):
    """Le planner n'a pas pu produire un plan exploitable."""


# ----------------------------------------------------------------------
# Contrat
# ----------------------------------------------------------------------
class ChunkPlanner(ABC):
    """Contrat commun à tous les planners de chunking."""

    @abstractmethod
    def plan(
        self,
        blocks: Sequence[Block],
        *,
        context_blocks: Sequence[Block] = (),
        feedback: str | None = None,
    ) -> ChunkPlan:
        """Retourne un plan couvrant exactement les blocs indexables de `blocks`.

        `context_blocks` : blocs précédents fournis en lecture seule (titres
        ouverts avant ce lot) ; ils ne doivent pas être replanifiés mais peuvent
        apparaître dans un `section_path`.
        `feedback` : erreurs de validation de la tentative précédente, à corriger.
        """


# ----------------------------------------------------------------------
# Schéma JSON strict (sortie structurée)
# ----------------------------------------------------------------------
CHUNK_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chunks"],
    "properties": {
        "chunks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "block_ids", "section_path"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Identifiant du chunk, ex. C001.",
                    },
                    "block_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                        "description": (
                            "block_id existants, consécutifs, dans l'ordre du document."
                        ),
                    },
                    "section_path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "block_id des titres englobants, du plus général au plus "
                            "précis. Uniquement des blocs de type heading."
                        ),
                    },
                },
            },
        }
    },
}


SYSTEM_PROMPT = """Tu es un planificateur de découpage documentaire (semantic chunking).

Tu reçois la liste ordonnée des blocs d'un document, chacun identifié par un \
block_id. Ta seule tâche est de décider quels blocs consécutifs forment une unité \
sémantique cohérente, et sous quels titres ils se trouvent.

INTERDICTIONS ABSOLUES :
- Ne génère JAMAIS de texte de contenu : tu ne renvoies que des identifiants.
- N'invente aucun contenu, ne paraphrase pas, ne résume pas, ne traduis pas.
- Ne corrige pas le texte, même s'il contient des fautes.
- Ne supprime aucun bloc : chaque block_id de la liste doit apparaître une fois.
- Ne duplique aucun block_id.
- Ne réordonne pas les blocs : l'ordre de la liste doit être conservé à l'identique.
- N'invente aucun block_id : n'utilise que ceux fournis.

RÈGLES DE DÉCOUPAGE :
- Un chunk est une suite de blocs CONSÉCUTIFS.
- Coupe aux frontières de sens : changement de sujet, nouveau titre, nouvelle \
section, nouvelle question traitée.
- Un titre doit être rattaché au contenu qu'il introduit, jamais laissé seul.
- Une liste, un tableau ou un bloc de code reste avec le paragraphe qui l'introduit.
- Vise environ {target} caractères par chunk, sans dépasser {maximum} caractères \
(la taille de chaque bloc t'est indiquée). Mieux vaut un chunk un peu court qu'un \
chunk qui mélange deux sujets.
- section_path contient les block_id de TOUS les titres qui gouvernent le chunk, du \
plus général au plus précis : le titre du document, puis les sous-titres ouverts, \
Y COMPRIS le titre placé en tête du chunk lui-même. Liste vide seulement si aucun \
titre ne précède le chunk.
  Exemple : si B001 est le titre du document, B012 la section « Architecture » et \
B013 la sous-section « Backend » placée en tête du chunk, alors \
section_path = ["B001", "B012", "B013"].
- Numérote les chunks C001, C002, ... dans l'ordre.

Réponds UNIQUEMENT par un objet JSON conforme au schéma imposé."""


USER_PROMPT_TEMPLATE = """Document : {source}

{context}Blocs à planifier ({count}) :
{blocks}

Découpe ces blocs en chunks sémantiques.

Ta réponse doit contenir EXACTEMENT ces {count} block_id, chacun une seule fois, \
dans cet ordre, et aucun autre :
{ids}

Tu ne renvoies aucun texte de contenu.{feedback}"""


# ----------------------------------------------------------------------
# Base commune aux planners pilotés par un LLM
# ----------------------------------------------------------------------
class LLMChunkPlanner(ChunkPlanner):
    """Construit le prompt, appelle un modèle, relit sa réponse.

    Les sous-classes n'implémentent que `_complete()` : tout ce qui touche au
    contrat (prompt, schéma, lecture de la réponse) est mutualisé, pour que
    Mistral et un modèle local se comportent exactement pareil.
    """

    def __init__(
        self,
        model: str,
        target_chunk_chars: int = 1200,
        max_chunk_chars: int = 2000,
        preview_chars: int = 400,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._target = target_chunk_chars
        self._maximum = max_chunk_chars
        self._preview = preview_chars
        self._temperature = temperature

    # ------------------------------------------------------------------
    def plan(
        self,
        blocks: Sequence[Block],
        *,
        context_blocks: Sequence[Block] = (),
        feedback: str | None = None,
    ) -> ChunkPlan:
        if not blocks:
            return ChunkPlan(chunks=[])

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(target=self._target, maximum=self._maximum),
            },
            {
                "role": "user",
                "content": self._build_prompt(blocks, context_blocks, feedback),
            },
        ]

        try:
            raw = self._complete(messages, structured=True)
        except PlannerError:
            # Panne franche (serveur injoignable) : inutile de retenter sans schéma.
            raise
        except Exception as error:  # noqa: BLE001
            # Certains modèles/versions d'API refusent le schéma : repli JSON libre.
            logger.warning("Sortie structurée indisponible (%s) — repli sur JSON libre.", error)
            raw = self._complete(messages, structured=False)

        return self._parse(raw)

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]], *, structured: bool) -> str:
        """Envoie les messages au modèle et retourne le texte brut de sa réponse."""

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        blocks: Sequence[Block],
        context_blocks: Sequence[Block],
        feedback: str | None,
    ) -> str:
        context = ""
        headings = [b for b in context_blocks if b.is_heading]
        if headings:
            lignes = "\n".join(self._describe(b) for b in headings[-6:])
            context = (
                "Titres déjà ouverts dans le document (contexte, à NE PAS replanifier ; "
                "utilisables dans section_path) :\n" + lignes + "\n\n"
            )

        return USER_PROMPT_TEMPLATE.format(
            source=blocks[0].source,
            context=context,
            count=len(blocks),
            blocks="\n".join(self._describe(b) for b in blocks),
            # Rappel explicite : un petit modèle a tendance à replanifier le contexte.
            ids=", ".join(b.block_id for b in blocks),
            feedback=(
                "\n\nLa tentative précédente a été REJETÉE par le validateur :\n"
                f"{feedback}\nCorrige exactement ces erreurs."
                if feedback
                else ""
            ),
        )

    def _describe(self, block: Block) -> str:
        text = " ".join(block.text.split())
        if len(text) > self._preview:
            text = text[: self._preview] + " […]"
        niveau = f" n{block.level}" if block.is_heading else ""
        return f"[{block.block_id}] {block.type}{niveau} ({len(block.text)} car.) : {text}"

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw: str) -> ChunkPlan:
        """Convertit la réponse brute du LLM en `ChunkPlan` (sans rien valider)."""
        text = raw.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise PlannerError(f"Réponse du LLM illisible (JSON invalide) : {error}") from error

        if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
            raise PlannerError("Réponse du LLM : clé 'chunks' absente ou mal typée.")

        chunks: list[PlannedChunk] = []
        for position, item in enumerate(payload["chunks"], start=1):
            if not isinstance(item, dict):
                raise PlannerError(f"Chunk n°{position} : objet attendu.")
            block_ids = item.get("block_ids")
            section_path = item.get("section_path") or []
            if not isinstance(block_ids, list) or not all(isinstance(i, str) for i in block_ids):
                raise PlannerError(f"Chunk n°{position} : 'block_ids' doit être une liste de chaînes.")
            if not isinstance(section_path, list) or not all(
                isinstance(i, str) for i in section_path
            ):
                raise PlannerError(
                    f"Chunk n°{position} : 'section_path' doit être une liste de chaînes."
                )
            chunks.append(
                PlannedChunk(
                    id=str(item.get("id") or f"C{position:03d}"),
                    block_ids=list(block_ids),
                    section_path=list(section_path),
                )
            )

        return ChunkPlan(chunks=chunks)



class MistralChunkPlanner(LLMChunkPlanner):
    """Planner s'appuyant sur l'API chat de Mistral, en sortie JSON contrainte."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        target_chunk_chars: int = 1200,
        max_chunk_chars: int = 2000,
        preview_chars: int = 400,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(model, target_chunk_chars, max_chunk_chars, preview_chars, temperature)
        if client is None:
            from mistralai import Mistral  # import tardif : facilite les tests

            client = Mistral(api_key=api_key)
        self._client = client

    # ------------------------------------------------------------------
    def _complete(self, messages: list[dict[str, str]], *, structured: bool) -> str:
        response_format: dict[str, Any] = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_plan",
                    "schema": CHUNK_PLAN_SCHEMA,
                    "strict": True,
                },
            }
            if structured
            else {"type": "json_object"}
        )

        response = retry(
            lambda: self._client.chat.complete(
                model=self._model,
                temperature=self._temperature,
                messages=messages,
                response_format=response_format,
            ),
            description="l'appel au planner Mistral",
        )
        return response.choices[0].message.content or ""


# ----------------------------------------------------------------------
# Planner local (Ollama)
# ----------------------------------------------------------------------
class OllamaChunkPlanner(LLMChunkPlanner):
    """Planner s'appuyant sur un modèle local servi par Ollama.

    Le découpage est l'étage le plus bavard du pipeline (un appel par lot de
    blocs) : le confier à un modèle local évite d'y consommer des crédits, tout
    en gardant Mistral pour la réponse finale.

    L'API `/api/chat` d'Ollama accepte un schéma JSON dans le champ `format` :
    on lui impose le même `CHUNK_PLAN_SCHEMA` qu'à Mistral. Le dialogue passe par
    `urllib` (bibliothèque standard) : aucune dépendance supplémentaire.

    Un modèle local se trompe plus souvent qu'un grand modèle — c'est sans risque
    pour la fidélité : le validateur rejette, on réessaie, puis on retombe sur le
    plan déterministe.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        url: str = "http://localhost:11434",
        target_chunk_chars: int = 1200,
        max_chunk_chars: int = 2000,
        preview_chars: int = 400,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        timeout: float = 180.0,
        attempts: int = 4,
        transport: Any | None = None,
    ) -> None:
        super().__init__(model, target_chunk_chars, max_chunk_chars, preview_chars, temperature)
        self._url = url.rstrip("/")
        self._num_ctx = num_ctx
        self._timeout = timeout
        self._attempts = attempts
        # `transport` : point d'injection pour les tests (payload -> réponse JSON).
        self._transport = transport or self._post

    # ------------------------------------------------------------------
    def _complete(self, messages: list[dict[str, str]], *, structured: bool) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                # Un lot de blocs est long : sans fenêtre élargie, le modèle ne
                # voit que la fin du prompt et invente des block_id.
                "num_ctx": self._num_ctx,
            },
        }
        # Sortie structurée : schéma complet, ou simple contrainte "du JSON".
        payload["format"] = CHUNK_PLAN_SCHEMA if structured else "json"

        try:
            response = retry(
                lambda: self._transport(payload),
                attempts=self._attempts,
                description=f"l'appel au planner Ollama ({self._model})",
            )
        except RuntimeError as error:
            # Ollama muet ou éteint : on remonte une erreur de planner, que le
            # SemanticChunker traduira en repli déterministe.
            raise PlannerError(
                f"Ollama ({self._model}) n'a pas répondu sur {self._url} : {error}. "
                "Le serveur est-il démarré (`ollama serve`) et le modèle installé "
                f"(`ollama pull {self._model}`) ?"
            ) from error
        return (response.get("message") or {}).get("content") or ""

    # ------------------------------------------------------------------
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/chat, sans dépendance externe."""
        requete = urllib.request.Request(
            f"{self._url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(requete, timeout=self._timeout) as reponse:
            return json.loads(reponse.read().decode("utf-8"))


# ----------------------------------------------------------------------
# Planner déterministe (repli)
# ----------------------------------------------------------------------
class HeuristicChunkPlanner(ChunkPlanner):
    """Plan déterministe : regroupement par section, borné en taille.

    Règles :
      - un titre ouvre un nouveau chunk et lui est rattaché ;
      - on accumule les blocs suivants tant que la taille cible n'est pas dépassée ;
      - un titre de niveau inférieur ou égal ferme la section courante.

    Ce planner n'appelle aucune API : il sert de repli garanti et de mode
    « sans LLM ». Sa sortie passe par le même validateur.
    """

    def __init__(self, target_chunk_chars: int = 1200, max_chunk_chars: int = 2000) -> None:
        self._target = target_chunk_chars
        self._maximum = max_chunk_chars

    def plan(
        self,
        blocks: Sequence[Block],
        *,
        context_blocks: Sequence[Block] = (),
        feedback: str | None = None,
    ) -> ChunkPlan:
        indexables = [b for b in blocks if b.indexable]
        if not indexables:
            return ChunkPlan(chunks=[])

        stack: list[Block] = [b for b in context_blocks if b.is_heading]
        chunks: list[PlannedChunk] = []
        current: list[Block] = []
        current_path: list[str] = [b.block_id for b in stack]
        size = 0

        def flush() -> None:
            nonlocal current, size
            if current:
                chunks.append(
                    PlannedChunk(
                        id=f"C{len(chunks) + 1:03d}",
                        block_ids=[b.block_id for b in current],
                        section_path=list(current_path),
                    )
                )
                current = []
                size = 0

        for block in indexables:
            if block.is_heading:
                # Un nouveau titre ferme la section précédente...
                flush()
                while stack and stack[-1].level >= block.level:
                    stack.pop()
                # ...puis devient le titre courant (il est inclus dans son chunk).
                current_path = [b.block_id for b in stack] + [block.block_id]
                stack.append(block)
            elif size and size + len(block.text) > self._target:
                # Section trop longue : on continue sous le même section_path.
                flush()

            current.append(block)
            size += len(block.text)

            if size >= self._maximum:
                flush()

        flush()
        return ChunkPlan(chunks=chunks)
