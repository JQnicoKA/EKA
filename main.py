"""Interface en ligne de commande (alternative à l'interface Streamlit).

Exemples :
    python main.py ingest data/             # indexe tous les .txt d'un dossier
    python main.py ingest data/ -r          # ... y compris les sous-dossiers
    python main.py ingest data/notes.txt    # indexe un seul fichier
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG minimaliste (Mistral + Qdrant).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Indexer un .txt ou un dossier de .txt.")
    p_ingest.add_argument("path", type=Path, help="Chemin du fichier ou du dossier.")
    p_ingest.add_argument("-r", "--recursive", action="store_true",
                          help="Descendre dans les sous-dossiers.")
    p_ingest.add_argument("--resume", action="store_true",
                          help="Sauter les documents déjà indexés.")

    p_ask = sub.add_parser("ask", help="Poser une question aux documents indexés.")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--top-k", type=int, default=None, help="Nombre de chunks à récupérer.")

    sub.add_parser("status", help="Afficher l'état de la base vectorielle.")
    sub.add_parser("reset", help="Vider la collection Qdrant.")

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        settings = Settings.from_env()
        pipeline = RAGPipeline.from_settings(settings)
    except Exception as error:  # noqa: BLE001
        print(f"Erreur de démarrage : {error}", file=sys.stderr)
        return 1

    if args.command == "ingest":
        if args.path.is_dir():
            results = pipeline.ingest_directory(
                args.path, recursive=args.recursive, resume=args.resume
            )
            if not results:
                print(f"Aucun fichier .txt trouvé dans {args.path}.")
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
            print(f"  [{position}] {result.chunk.reference} (score {result.score:.3f})")
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
