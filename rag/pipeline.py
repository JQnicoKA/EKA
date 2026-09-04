"""Le chef d'orchestre : assemble tous les composants du RAG.

`RAGPipeline` ne contient aucune logique métier propre ; il se contente
d'enchaîner les composants. Chaque dépendance est injectée dans le constructeur
(*dependency injection*), ce qui rend la classe testable : on peut remplacer
n'importe quel étage par un faux objet dans les tests.

Deux flux :
  - INGESTION : .txt -> blocs -> plan validé -> chunks -> embeddings -> Qdrant
  - INTERROGATION : question -> embedding -> recherche -> prompt -> LLM -> réponse
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Settings
from .embedder import Embedder, MistralEmbedder
from .generator import AnswerGenerator, MistralGenerator
from .models import Answer, Chunk
from .parser import DocumentParser, TextParser
from .planner import (
    ChunkPlanner,
    HeuristicChunkPlanner,
    MistralChunkPlanner,
    OllamaChunkPlanner,
)
from .reconstructor import ChunkReconstructor
from .semantic_chunker import SemanticChunker
from .validator import PlanValidator
from .vector_store import QdrantVectorStore, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Point d'entrée unique de l'application."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: SemanticChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: AnswerGenerator,
        top_k: int = 4,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Fabrique : construit un pipeline complet à partir de la configuration
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Assemble les implémentations concrètes (Mistral + Qdrant)."""
        embedder = MistralEmbedder(
            api_key=settings.mistral_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )

        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
            vector_size=embedder.dimension,
            score_threshold=settings.score_threshold,
        )

        return cls(
            parser=TextParser(max_block_chars=settings.max_block_chars),
            chunker=cls.build_chunker(settings),
            embedder=embedder,
            vector_store=vector_store,
            generator=MistralGenerator(
                api_key=settings.mistral_api_key,
                model=settings.llm_model,
            ),
            top_k=settings.top_k,
        )

    @staticmethod
    def build_chunker(settings: Settings) -> SemanticChunker:
        """Choisit le planner (LLM ou déterministe) et câble le chunker."""
        fallback = HeuristicChunkPlanner(
            target_chunk_chars=settings.target_chunk_chars,
            max_chunk_chars=settings.max_chunk_chars,
        )

        planner: ChunkPlanner
        if settings.chunking_mode != "semantic":
            planner = fallback
        elif settings.planner_provider == "ollama":
            # Modèle local : le découpage ne consomme aucun crédit.
            planner = OllamaChunkPlanner(
                model=settings.planner_model or settings.ollama_model,
                url=settings.ollama_url,
                target_chunk_chars=settings.target_chunk_chars,
                max_chunk_chars=settings.max_chunk_chars,
                num_ctx=settings.ollama_num_ctx,
                timeout=settings.ollama_timeout,
            )
        else:
            planner = MistralChunkPlanner(
                api_key=settings.mistral_api_key,
                model=settings.planner_model or settings.llm_model,
                target_chunk_chars=settings.target_chunk_chars,
                max_chunk_chars=settings.max_chunk_chars,
            )

        return SemanticChunker(
            planner=planner,
            validator=PlanValidator(),
            reconstructor=ChunkReconstructor(),
            fallback_planner=fallback,
            target_chunk_chars=settings.target_chunk_chars,
            max_chunk_chars=settings.max_chunk_chars,
            min_chunk_chars=settings.min_chunk_chars,
            batch_blocks=settings.planner_batch_blocks,
            max_attempts=settings.planner_max_attempts,
        )

    # ------------------------------------------------------------------
    # Flux 1 : ingestion
    # ------------------------------------------------------------------
    def ingest_file(self, path: Path) -> int:
        """Indexe un document et retourne le nombre de chunks stockés."""
        path = Path(path)
        logger.info("Ingestion de %s...", path.name)

        document = self.parser.parse(path)              # 1. blocs identifiés
        chunks: list[Chunk] = self.chunker.chunk(document)  # 2. plan validé -> chunks
        if not chunks:
            return 0

        # Le document est réécrit intégralement : on supprime d'abord ses anciens
        # chunks, sinon un document raccourci laisserait des points orphelins.
        self.vector_store.delete_source(document.source)

        vectors = self.embedder.embed_documents([c.text for c in chunks])  # 3. embeddings
        return self.vector_store.add(chunks, vectors)                      # 4. stockage

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = False,
        batch_chunks: int = 256,
        resume: bool = False,
    ) -> dict[str, int]:
        """Indexe tous les fichiers texte d'un dossier. Retourne {nom: nb_chunks}.

        Les chunks sont accumulés **entre documents** avant d'être vectorisés :
        un document ne pesant que ~6 chunks, les envoyer un par un multiplierait
        par cinq le nombre d'appels à l'API d'embeddings.

        `resume=True` saute les documents déjà présents dans la base. C'est sans
        risque : les chunks d'un document sont toujours écrits en une seule fois,
        donc un document indexé l'est intégralement. Une ingestion interrompue se
        reprend là où elle s'est arrêtée, sans réindexer ni dupliquer.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {directory}")

        paths = sorted(
            path
            for path in (directory.rglob("*") if recursive else directory.glob("*"))
            if path.is_file() and self.parser.supports(path)
        )

        if resume:
            deja_indexes = set(self.vector_store.list_sources())
            restants = [p for p in paths if p.name not in deja_indexes]
            logger.info(
                "Reprise : %d document(s) déjà indexé(s), %d à traiter.",
                len(paths) - len(restants), len(restants),
            )
            paths = restants

        results: dict[str, int] = {}
        tampon: list[Chunk] = []

        def vider() -> None:
            """Vectorise et stocke le tampon ; en cas d'échec, marque les documents à 0."""
            if not tampon:
                return
            try:
                vectors = self.embedder.embed_documents([c.text for c in tampon])
                self.vector_store.add(tampon, vectors)
            except Exception as error:  # noqa: BLE001
                sources = sorted({c.source for c in tampon})
                logger.error("Échec du lot d'embeddings (%d chunk(s)) : %s", len(tampon), error)
                for source in sources:
                    results[source] = 0
            tampon.clear()

        for position, path in enumerate(paths, start=1):
            try:
                document = self.parser.parse(path)
                chunks = self.chunker.chunk(document)
            except Exception as error:  # noqa: BLE001
                # Un document illisible ne doit pas interrompre tout le lot.
                logger.error("Échec de l'ingestion de %s : %s", path.name, error)
                results[path.name] = 0
                continue

            # Le document est réécrit : ses anciens chunks disparaissent d'abord.
            self.vector_store.delete_source(document.source)
            results[path.name] = len(chunks)
            tampon.extend(chunks)

            if len(tampon) >= batch_chunks:
                vider()
            if position % 500 == 0:
                logger.info("Ingestion : %d/%d document(s)", position, len(paths))

        vider()
        return results

    # ------------------------------------------------------------------
    # Flux 2 : interrogation
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """Répond à une question en s'appuyant sur les documents indexés."""
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide.")

        query_vector = self.embedder.embed_query(question)               # 5. embedding
        contexts = self.vector_store.search(query_vector, top_k or self.top_k)  # 6. recherche
        text = self.generator.generate(question, contexts)               # 7. prompt + LLM

        return Answer(question=question, text=text, sources=contexts)
