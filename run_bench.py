"""Exécute le RAG sur les questions d'EnterpriseRAG-Bench et produit le fichier de réponses.

Le benchmark attend un JSONL dont chaque ligne vaut :

    {"question_id": "qst_0001", "answer": "...", "document_ids": ["dsid_ab...", ...]}

Ce script ne fait *que* produire ce fichier. Le calcul des quatre métriques
(correctness, completeness, document recall, invalid extra documents) est
ensuite réalisé par le dépôt EnterpriseRAG-Bench, qui exige une clé OpenAI ou
Anthropic pour son LLM juge.

Exemples :
    # essai sur 10 questions
    python run_bench.py --limit 10

    # passage complet, en reprenant si interrompu
    python run_bench.py --resume

    # plus de documents candidats (utile pour les questions "completeness")
    python run_bench.py --top-k 30 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rag import RAGPipeline, Settings
from rag.embedder import MistralEmbedder
from rag.generator import MistralGenerator
from rag.models import Answer
from rag.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Le corpus et les réponses de référence du benchmark sont en anglais : demander
# une réponse en français ferait chuter le score de correctness pour une raison
# qui n'a rien à voir avec la qualité du RAG.
SYSTEM_PROMPT_EN = """You are a document assistant. Answer the user's question \
based EXCLUSIVELY on the provided document excerpts.

Rules:
- Do not use any knowledge beyond the provided excerpts.
- If the excerpts do not contain the answer, say so plainly: \
"The provided documents do not contain the answer to this question."
- Answer in English, concisely and factually.
- Give the answer itself; do not describe the excerpts."""


def doc_id_depuis_source(source: str) -> str:
    """Extrait l'identifiant du benchmark depuis un nom de fichier.

    "dsid_ae068ee4...__multipart-upload-limits.txt" -> "dsid_ae068ee4..."
    """
    return source.split("__", 1)[0]


def documents_candidats(reponse: Answer, maximum: int) -> list[str]:
    """Documents distincts touchés par les chunks retrouvés, dans l'ordre des scores.

    Plusieurs chunks proviennent souvent du même document : on déduplique en
    conservant le meilleur rang de chacun, car le benchmark raisonne en
    documents, pas en passages.
    """
    vus: list[str] = []
    for resultat in reponse.sources:
        doc = doc_id_depuis_source(resultat.chunk.source)
        if doc not in vus:
            vus.append(doc)
        if len(vus) >= maximum:
            break
    return vus


def construire_pipeline(settings: Settings, anglais: bool) -> RAGPipeline:
    """Pipeline standard, avec éventuellement la consigne en anglais."""
    embedder = MistralEmbedder(
        api_key=settings.mistral_api_key,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
        vector_size=embedder.dimension,
        score_threshold=settings.score_threshold,
    )
    generateur = MistralGenerator(
        api_key=settings.mistral_api_key,
        model=settings.llm_model,
        system_prompt=SYSTEM_PROMPT_EN if anglais else None,
    )
    # Le loader et le chunker ne servent pas ici : on interroge un index existant.
    return RAGPipeline(
        loader=None,
        chunker=None,
        embedder=embedder,
        vector_store=store,
        generator=generateur,
        top_k=settings.top_k,
    )


def charger_questions(chemin: Path) -> list[dict]:
    with chemin.open() as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def deja_traitees(chemin: Path) -> set[str]:
    """Identifiants déjà présents dans le fichier de sortie (pour --resume)."""
    if not chemin.exists():
        return set()
    faits: set[str] = set()
    with chemin.open() as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                faits.add(json.loads(ligne)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return faits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path,
                        default=Path("../EnterpriseRAG-Bench-main/questions.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("../EnterpriseRAG-Bench-main/answer_evaluation/answers.jsonl"))
    parser.add_argument("--top-k", type=int, default=20,
                        help="Chunks récupérés par question (défaut 20). Les questions "
                             "'completeness' attendent jusqu'à 10 documents distincts.")
    parser.add_argument("--max-docs", type=int, default=10,
                        help="Documents distincts déclarés au maximum (défaut 10).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ne traiter que les N premières questions.")
    parser.add_argument("--resume", action="store_true",
                        help="Sauter les questions déjà présentes dans le fichier de sortie.")
    parser.add_argument("--francais", action="store_true",
                        help="Garder la consigne en français (par défaut : anglais, "
                             "comme les réponses de référence).")
    args = parser.parse_args()

    try:
        settings = Settings.from_env()
        pipeline = construire_pipeline(settings, anglais=not args.francais)
    except Exception as error:  # noqa: BLE001
        print(f"Erreur de démarrage : {error}", file=sys.stderr)
        return 1

    if not args.questions.exists():
        print(f"Fichier de questions introuvable : {args.questions}", file=sys.stderr)
        return 1

    questions = charger_questions(args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    faites = deja_traitees(args.output) if args.resume else set()
    restantes = [q for q in questions if q["question_id"] not in faites]
    if args.limit is not None:
        restantes = restantes[: args.limit]

    print(f"collection : {settings.collection_name}")
    print(f"modèle     : {settings.llm_model} | top_k : {args.top_k}")
    print(f"questions  : {len(questions)} au total, {len(faites)} déjà faites, "
          f"{len(restantes)} à traiter")
    if not restantes:
        print("rien à faire.")
        return 0

    debut = time.time()
    echecs = 0

    # Ouverture en ajout : chaque réponse est écrite immédiatement, donc une
    # interruption ne perd que la question en cours.
    with args.output.open("a") as sortie:
        for position, question in enumerate(restantes, start=1):
            qid = question["question_id"]
            try:
                reponse = pipeline.ask(question["question"], top_k=args.top_k)
                ligne = {
                    "question_id": qid,
                    "answer": reponse.text,
                    "document_ids": documents_candidats(reponse, args.max_docs),
                }
            except Exception as error:  # noqa: BLE001
                echecs += 1
                print(f"  [ÉCHEC] {qid} : {error}", file=sys.stderr)
                continue

            sortie.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            sortie.flush()

            if position % 10 == 0 or position == len(restantes):
                ecoule = time.time() - debut
                reste = ecoule / position * (len(restantes) - position)
                print(f"  {position}/{len(restantes)} — {ecoule/60:.1f} min écoulées, "
                      f"~{reste/60:.1f} min restantes", flush=True)

    print(f"\nterminé en {(time.time()-debut)/60:.1f} min | échecs : {echecs}")
    print(f"fichier : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
