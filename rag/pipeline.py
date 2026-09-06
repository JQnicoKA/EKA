"""Le chef d'orchestre : assemble tous les composants du RAG.

`RAGPipeline` ne contient aucune logique métier propre ; il se contente
d'enchaîner les composants. Chaque dépendance est injectée dans le constructeur
(*dependency injection*), ce qui rend la classe testable : on peut remplacer
n'importe quel étage par un faux objet dans les tests.

Deux flux :
  - INGESTION : .txt -> chunks -> embeddings -> Qdrant
  - INTERROGATION : question -> embedding -> recherche -> prompt -> LLM -> réponse
"""

from __future__ import annotations

import logging
from pathlib import Path

from .chunker import TextChunker
from .config import Settings
from .embedder import Embedder, MistralEmbedder
from .generator import AnswerGenerator, MistralGenerator
from .loader import DocumentLoader, TextLoader
from .models import Answer, Chunk
from .vector_store import QdrantVectorStore, VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Point d'entrée unique de l'application."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: AnswerGenerator,
        top_k: int = 4,
    ) -> None:
        self.loader = loader
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
            loader=TextLoader(),
            chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
            embedder=embedder,
            vector_store=vector_store,
            generator=MistralGenerator(
                api_key=settings.mistral_api_key,
                model=settings.llm_model,
            ),
            top_k=settings.top_k,
        )

    # ------------------------------------------------------------------
    # Flux 1 : ingestion
    # ------------------------------------------------------------------
    def ingest_file(self, path: Path) -> int:
        """Indexe un document et retourne le nombre de chunks stockés."""
        path = Path(path)
        logger.info("Ingestion de %s...", path.name)

        document = self.loader.load(path)                    # 1. chargement
        chunks: list[Chunk] = self.chunker.split(document)   # 2. découpage
        if not chunks:
            return 0

        vectors = self.embedder.embed_documents([c.text for c in chunks])  # 3. embeddings
        return self.vector_store.add(chunks, vectors)                      # 4. stockage

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = False,
        batch_chunks: int = 256,
        resume: bool = False,
    ) -> dict[str, int]:
        """Indexe tous les .txt d'un dossier. Retourne {nom_fichier: nb_chunks}.

        `recursive=True` descend dans les sous-dossiers.

        Les chunks sont accumulés **entre documents** avant d'être vectorisés.
        Sans ce tampon, un document de 7 chunks déclencherait un appel réseau à
        lui seul ; en regroupant, on remplit les lots de 32 textes que
        `MistralEmbedder` envoie par requête. Sur un gros corpus cela divise le
        nombre d'allers-retours par cinq environ.

        `resume=True` saute les documents déjà présents dans la base. C'est sans
        risque : le tampon n'est vidé qu'entre deux documents, jamais au milieu
        de l'un d'eux, donc un document présent dans la base y est intégralement.
        Une ingestion interrompue se reprend là où elle s'est arrêtée.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {directory}")

        motif = "**/*.txt" if recursive else "*.txt"
        paths = sorted(directory.glob(motif))

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
            except Exception as error:  # noqa: BLE001
                # Échec de vectorisation : souvent transitoire (quota, réseau).
                # Un lot perdu ne doit pas interrompre le reste du corpus, mais
                # les documents concernés doivent apparaître comme non indexés.
                sources = sorted({c.source for c in tampon})
                logger.error("Échec du lot d'embeddings (%d chunk(s)) : %s", len(tampon), error)
                for source in sources:
                    results[source] = 0
                tampon.clear()
                return

            try:
                self.vector_store.add(tampon, vectors)
            except Exception as error:
                # Échec d'écriture : la base est injoignable ou en panne. Inutile
                # de continuer — poursuivre reviendrait à payer la vectorisation
                # de tout le reste du corpus pour la jeter. On arrête net ; une
                # relance avec `resume=True` repartira de ce qui est en base.
                tampon.clear()
                raise RuntimeError(
                    f"Écriture impossible dans la base vectorielle : {error}. "
                    "Ingestion interrompue ; relancez avec resume=True."
                ) from error

            tampon.clear()

        for position, path in enumerate(paths, start=1):
            try:
                document = self.loader.load(path)
                chunks = self.chunker.split(document)
            except Exception as error:  # noqa: BLE001
                # Un document illisible ne doit pas interrompre tout le lot.
                logger.error("Échec de l'ingestion de %s : %s", path.name, error)
                results[path.name] = 0
                continue

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
