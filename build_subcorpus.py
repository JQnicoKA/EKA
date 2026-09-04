"""Construit un sous-corpus exploitable à partir d'EnterpriseRAG-bench.

Le corpus complet (~512 000 fichiers) n'est pas indexable sur une machine de
développement : 3,2 M vecteurs, 625 M tokens d'embeddings, 13 Go de RAM côté
Qdrant. Or `questions.jsonl` ne référence que ~720 documents « attendus ».

Ce script assemble un dossier plat contenant :
  - **tous les documents attendus** par les questions (sinon aucune réponse
    n'est trouvable) ;
  - **N distracteurs** tirés au hasard, répartis proportionnellement aux
    sources du corpus (slack, gmail, jira...) pour que la recherche reste
    réaliste.

Exemples :
    python build_subcorpus.py                              # 722 gold + 20 000 distracteurs
    python build_subcorpus.py --distracteurs 5000 --liens  # plus léger, sans copie
    python main.py ingest bench                            # puis on indexe

Le dossier produit est plat : `main.py ingest` le traite sans `--recursive`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PREFIXE = "dsid_"
LONGUEUR_ID = 32  # dsid_<32 caractères hexadécimaux>__slug.txt


def indexer(corpus: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Parcourt le corpus une fois : {dsid: chemin} et {source: [dsid, ...]}."""
    par_id: dict[str, Path] = {}
    par_source: dict[str, list[str]] = defaultdict(list)

    for dossier, _, fichiers in os.walk(corpus):
        source = os.path.relpath(dossier, corpus).split(os.sep)[0]
        for nom in fichiers:
            if not nom.endswith(".txt") or not nom.startswith(PREFIXE):
                continue
            identifiant = nom[len(PREFIXE) : len(PREFIXE) + LONGUEUR_ID]
            par_id[identifiant] = Path(dossier) / nom
            par_source[source].append(identifiant)

    return par_id, dict(par_source)


def documents_attendus(questions: Path) -> set[str]:
    """Identifiants cités par les questions du benchmark."""
    attendus: set[str] = set()
    with questions.open(encoding="utf-8") as flux:
        for ligne in flux:
            ligne = ligne.strip()
            if not ligne:
                continue
            for brut in json.loads(ligne).get("expected_doc_ids") or []:
                attendus.add(brut[len(PREFIXE):] if brut.startswith(PREFIXE) else brut)
    return attendus


def tirer_distracteurs(
    par_source: dict[str, list[str]],
    exclus: set[str],
    total: int,
    graine: int,
) -> list[str]:
    """Tirage stratifié : chaque source garde son poids dans le sous-corpus."""
    aleatoire = random.Random(graine)
    disponibles = {
        source: sorted(set(ids) - exclus) for source, ids in par_source.items()
    }
    population = sum(len(ids) for ids in disponibles.values())
    if population <= total:
        return [identifiant for ids in disponibles.values() for identifiant in ids]

    # Quota proportionnel, puis distribution du reste aux sources les plus fournies.
    quotas = {
        source: min(len(ids), int(total * len(ids) / population))
        for source, ids in disponibles.items()
    }
    reste = total - sum(quotas.values())
    for source in sorted(disponibles, key=lambda s: -len(disponibles[s])):
        if reste <= 0:
            break
        marge = min(reste, len(disponibles[source]) - quotas[source])
        quotas[source] += marge
        reste -= marge

    tirage: list[str] = []
    for source, ids in disponibles.items():
        tirage.extend(aleatoire.sample(ids, quotas[source]))
    return tirage


def installer(chemins: list[Path], sortie: Path, liens: bool) -> int:
    """Copie (ou lie) les fichiers dans le dossier de sortie. Retourne les octets."""
    octets = 0
    for source in chemins:
        destination = sortie / source.name
        octets += source.stat().st_size
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if liens:
            destination.symlink_to(source.resolve())
        else:
            shutil.copy2(source, destination)
    return octets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=Path("../EnterpriseRAG-bench"))
    parser.add_argument("--questions", type=Path, default=None,
                        help="Défaut : <corpus>/questions.jsonl")
    parser.add_argument("--sortie", type=Path, default=Path("bench"))
    parser.add_argument("--distracteurs", type=int, default=20_000)
    parser.add_argument("--graine", type=int, default=0, help="Rend le tirage reproductible.")
    parser.add_argument("--liens", action="store_true",
                        help="Créer des liens symboliques au lieu de copier.")
    parser.add_argument("--vider", action="store_true",
                        help="Vider le dossier de sortie avant de le remplir.")
    args = parser.parse_args()

    questions = args.questions or args.corpus / "questions.jsonl"
    if not args.corpus.is_dir():
        print(f"Corpus introuvable : {args.corpus}")
        return 1
    if not questions.is_file():
        print(f"Fichier de questions introuvable : {questions}")
        return 1

    print(f"Parcours de {args.corpus}...")
    par_id, par_source = indexer(args.corpus)
    print(f"  {len(par_id):,} documents, {len(par_source)} sources")

    attendus = documents_attendus(questions)
    presents = sorted(i for i in attendus if i in par_id)
    manquants = sorted(attendus - set(presents))
    print(f"  {len(presents)} document(s) attendu(s) retrouvé(s)"
          + (f", {len(manquants)} introuvable(s)" if manquants else ""))

    distracteurs = tirer_distracteurs(par_source, set(presents), args.distracteurs, args.graine)
    retenus = presents + sorted(distracteurs)

    if args.vider and args.sortie.exists():
        shutil.rmtree(args.sortie)
    args.sortie.mkdir(parents=True, exist_ok=True)

    print(f"Installation de {len(retenus):,} fichiers dans {args.sortie}...")
    octets = installer([par_id[i] for i in retenus], args.sortie, args.liens)

    selection = set(retenus)
    repartition = Counter(
        source
        for source, ids in par_source.items()
        for identifiant in ids
        if identifiant in selection
    )
    manifeste = {
        "corpus": str(args.corpus),
        "graine": args.graine,
        "attendus": presents,
        "attendus_manquants": manquants,
        "distracteurs": len(distracteurs),
        "repartition": dict(repartition.most_common()),
        "liens_symboliques": args.liens,
    }
    (args.sortie / "manifeste.json").write_text(
        json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(questions, args.sortie / "questions.jsonl")

    print(f"\n{len(retenus):,} documents ({octets/1e6:.0f} Mo) — "
          f"{len(presents)} attendus + {len(distracteurs):,} distracteurs")
    for source, nombre in repartition.most_common():
        print(f"  {source:16s} {nombre:7,}")
    print(f"\nManifeste : {args.sortie / 'manifeste.json'}")
    print(f"Questions : {args.sortie / 'questions.jsonl'}")
    print(f"\nIndexation :  python main.py ingest {args.sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
