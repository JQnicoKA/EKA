"""Interface en ligne de commande (alternative à l'interface Streamlit).

Exemples :
    python main.py ingest data/            # indexe tous les .txt d'un dossier
    python main.py ingest data/rapport.txt # indexe un seul fichier
    python main.py chunks data/rapport.txt # inspecte le découpage sans rien indexer
    python main.py ask "Quel est le budget prévu ?"
    python main.py status                  # état de la base vectorielle
    python main.py reset                   # vide la base
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag import RAGPipeline, Settings
from rag.parser import TextParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG minimaliste (Mistral + Qdrant).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Indexer un .txt ou un dossier de .txt.")
    p_ingest.add_argument("path", type=Path, help="Chemin du fichier ou du dossier.")
    p_ingest.add_argument("--recursive", action="store_true", help="Parcourir les sous-dossiers.")
    p_ingest.add_argument("--reprendre", action="store_true",
                          help="Sauter les documents déjà indexés (reprise après interruption).")

    p_chunks = sub.add_parser(
        "chunks", help="Afficher le découpage d'un document sans l'indexer."
    )
    p_chunks.add_argument("path", type=Path)
    p_chunks.add_argument("--blocks", action="store_true", help="Afficher aussi les blocs.")

    p_ask = sub.add_parser("ask", help="Poser une question aux documents indexés.")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--top-k", type=int, default=None, help="Nombre de chunks à récupérer.")

    sub.add_parser("status", help="Afficher l'état de la base vectorielle.")
    sub.add_parser("reset", help="Vider la collection Qdrant.")

    return parser


def show_chunks(settings: Settings, args: argparse.Namespace) -> int:
    """Affiche le découpage d'un document sans rien indexer."""
    parser = TextParser(max_block_chars=settings.max_block_chars)
    chunker = RAGPipeline.build_chunker(settings)

    document = parser.parse(args.path)
    print(f"{document.source} : {len(document.blocks)} bloc(s), "
          f"{len(document.indexable_blocks)} indexable(s)")

    if args.blocks:
        print("\nBlocs :")
        for block in document.blocks:
            extrait = " ".join(block.text.split())[:70]
            print(f"  {block.block_id} {block.type:10s} l.{block.line_start:<4d} "
                  f"{len(block.text):5d} car. « {extrait} »")

    chunks = chunker.chunk(document)
    print(f"\n{len(chunks)} chunk(s) — mode « {settings.chunking_mode} » :")
    for chunk in chunks:
        extrait = " ".join(chunk.text.split())[:100]
        print(f"\n  {chunk.chunk_id} — {chunk.reference} — {len(chunk.text)} car.")
        print(f"      section : {chunk.section or '(racine)'}")
        print(f"      blocs   : {', '.join(chunk.block_ids)}")
        print(f"      « {extrait}... »")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    try:
        settings = Settings.from_env()
        # `chunks` n'inspecte qu'un fichier : inutile d'exiger Qdrant.
        if args.command == "chunks":
            return show_chunks(settings, args)
        pipeline = RAGPipeline.from_settings(settings)
    except Exception as error:  # noqa: BLE001
        print(f"Erreur de démarrage : {error}", file=sys.stderr)
        return 1

    if args.command == "ingest":
        if args.path.is_dir():
            results = pipeline.ingest_directory(
                args.path, recursive=args.recursive, resume=args.reprendre
            )
            if not results:
                print(f"Aucun fichier texte trouvé dans {args.path}.")
            for name, count in results.items():
                print(f"  {name} : {count} chunk(s)")
        else:
            count = pipeline.ingest_file(args.path)
            print(f"  {args.path.name} : {count} chunk(s)")

    elif args.command == "ask":
        answer = pipeline.ask(args.question, top_k=args.top_k)
        print(f"\nQuestion : {answer.question}\n")
        print(f"Réponse :\n{answer.text}\n")
        print("Sources :")
        for position, result in enumerate(answer.sources, start=1):
            extrait = " ".join(result.chunk.text.split())[:200]
            section = f" — {result.chunk.section}" if result.chunk.section else ""
            print(f"  [{position}] {result.chunk.reference}{section} (score {result.score:.3f})")
            print(f"      « {extrait}... »")

    elif args.command == "status":
        print(f"Collection : {settings.collection_name}")
        print(f"Chunks indexés : {pipeline.vector_store.count()}")
        print("Documents :")
        for source in pipeline.vector_store.list_sources():
            print(f"  • {source}")

    elif args.command == "reset":
        pipeline.vector_store.reset()
        print("Collection vidée.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
