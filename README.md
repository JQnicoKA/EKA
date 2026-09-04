# RAG minimaliste — Mistral + Qdrant

Un pipeline **Retrieval-Augmented Generation** réduit à l'essentiel : on indexe des
documents texte dans une base vectorielle, puis on interroge un LLM en lui fournissant
uniquement les passages pertinents. Chaque réponse est accompagnée de sa source
(document + lignes + section).

La particularité du projet est son **chunking sémantique piloté par LLM** : le modèle
décide *où couper*, jamais *ce que contient* un chunk. Le texte indexé est toujours
reconstruit depuis le document original par du code déterministe, puis vérifié.

---

## Le pipeline en deux flux

```
INGESTION (une fois par document)
   .txt ──▶ blocs ──▶ plan de chunking ──▶ validation ──▶ chunks ──▶ embeddings ──▶ Qdrant
         parser      planner (LLM)       validator    reconstructor  MistralEmbedder  vector_store

INTERROGATION (à chaque question)
   question ──▶ embedding ──▶ recherche top-k ──▶ prompt ──▶ LLM ──▶ réponse + sources
             MistralEmbedder    vector_store    generator   Mistral
```

## Architecture du code

| Fichier | Rôle | Classe principale |
|---|---|---|
| `rag/config.py` | Configuration lue depuis `.env` | `Settings` |
| `rag/models.py` | Objets métier échangés entre étages | `Block`, `ChunkPlan`, `Chunk`, `Answer` |
| `rag/parser.py` | 1. Document → blocs identifiés (`B001`…) | `TextParser` |
| `rag/planner.py` | 2. Blocs → plan de découpage (aucun texte) | `MistralChunkPlanner`, `OllamaChunkPlanner`, `HeuristicChunkPlanner` |
| `rag/validator.py` | 3. Contrôle strict du plan et des chunks | `PlanValidator` |
| `rag/reconstructor.py` | 4. Plan + document → texte des chunks | `ChunkReconstructor` |
| `rag/semantic_chunker.py` | Orchestration (lots, retry, repli) | `SemanticChunker` |
| `rag/embedder.py` | 5. Vectorisation | `MistralEmbedder` |
| `rag/vector_store.py` | 6. Stockage + recherche | `QdrantVectorStore` |
| `rag/generator.py` | 7. Prompt + appel au LLM | `MistralGenerator` |
| `rag/pipeline.py` | Assemblage de tous les étages | `RAGPipeline` |
| `app.py` / `main.py` | Interface web / ligne de commande | — |
| `build_subcorpus.py` | Extrait un sous-corpus exploitable du benchmark | — |
| `smoke_test.py` | Essai complet sans appeler l'API Mistral | — |
| `tests/` | Tests unitaires (`unittest`, sans dépendance) | — |

Chaque étage possède une **classe abstraite** (`DocumentParser`, `ChunkPlanner`,
`Embedder`, `VectorStore`, `AnswerGenerator`) et une implémentation concrète. Le
pipeline ne dépend que des abstractions : remplacer Qdrant par autre chose, ou Mistral
par un autre fournisseur, revient à écrire une nouvelle classe sans toucher au reste.

**Formats acceptés** : `.txt`, `.md` (pas de PDF — hors périmètre).

---

## Le chunking sémantique en détail

### Principe

Le LLM est un **planificateur**, pas un rédacteur. Il reçoit la liste des blocs du
document et ne renvoie que des identifiants :

```json
{
  "chunks": [
    { "id": "C001", "block_ids": ["B001", "B002"], "section_path": ["B001"] },
    { "id": "C002", "block_ids": ["B003", "B004"], "section_path": ["B001", "B003"] }
  ]
}
```

Le schéma JSON est imposé au modèle (`response_format: json_schema` côté Mistral,
champ `format` côté Ollama), et le prompt système lui interdit explicitement d'inventer,
paraphraser, résumer, corriger, supprimer, dupliquer ou réordonner quoi que ce soit.

### Qui planifie ? (`PLANNER_PROVIDER`)

| Valeur | Découpage | Coût | Remarque |
|---|---|---|---|
| `mistral` | API Mistral | crédits API | le plus fiable |
| `ollama` | modèle local (Ollama) | **gratuit** | plus lent, se trompe plus souvent — le validateur et le repli couvrent |
| *(`CHUNKING_MODE=heuristic`)* | aucun LLM | **gratuit** | instantané, purement structurel |

Dans tous les cas, la **réponse aux questions** et les **embeddings** restent chez
Mistral : seul l'étage de planification change.

```bash
# Découpage local, réponses Mistral
ollama serve            # si l'app Ollama n'est pas déjà lancée
ollama pull llama3.2:3b
# dans .env :
#   PLANNER_PROVIDER=ollama
#   OLLAMA_MODEL=llama3.2:3b
```

Un modèle local n'a aucun privilège : sa sortie subit les mêmes validations. S'il
échoue (ou si Ollama est éteint), le pipeline retente puis retombe sur le plan
déterministe — l'ingestion aboutit toujours.

### Garanties

| Priorité | Mécanisme |
|---|---|
| 1. Fidélité au document | Le texte d'un chunk est l'assemblage des blocs sources ; chaque bloc est vérifié contre `document.text[char_start:char_end]`. |
| 2. Aucune perte | Le validateur exige une couverture de 100 % des blocs indexables. |
| 3. Aucune hallucination | Tout `block_id` inconnu, dupliqué ou réordonné fait rejeter le plan. |
| 4. Traçabilité | Chaque chunk conserve ses `block_ids`, ses lignes et son `section_path`. |
| 5. Qualité sémantique | Frontières décidées par le LLM, bornées par `TARGET/MAX_CHUNK_CHARS`. |
| 6. Performance | Découpage en lots (`PLANNER_BATCH_BLOCKS`) ; mode `heuristic` sans appel API. |

Le `section_path` est proposé par le LLM sous forme de `block_id` de titres — donc
vérifiable — puis **recalé sur la hiérarchie réelle du document** s'il diverge : les
noms de sections proviennent toujours des titres du document.

### En cas de plan invalide

```
plan du LLM ──▶ validation ──✗──▶ nouvelle tentative (les erreurs sont renvoyées au LLM)
                          └──✗──▶ repli déterministe (HeuristicChunkPlanner)
                                           └──▶ validé lui aussi, sinon échec de l'ingestion
```

Aucun chunk invalide n'atteint les embeddings.

### Constituer un sous-corpus de travail

Le corpus `EnterpriseRAG-bench` (~512 000 fichiers) n'est pas indexable sur une machine
de développement : 3,2 M vecteurs, 625 M tokens d'embeddings, ~13 Go côté Qdrant. Or
`questions.jsonl` ne référence que **722 documents attendus**.

```bash
python build_subcorpus.py --distracteurs 20000    # -> bench/ (722 gold + 20 000 distracteurs)
python main.py ingest bench
```

Le tirage des distracteurs est stratifié par source (slack, gmail, jira…) et
reproductible (`--graine`). Le dossier produit contient un `manifeste.json` (documents
attendus, répartition, graine) et une copie de `questions.jsonl`. `--liens` crée des
liens symboliques au lieu de copier.

Budget mesuré pour 20 722 documents (129 358 chunks) :

| | Mode `semantic` (Mistral) | Mode `heuristic` |
|---|---|---|
| Découpage | 30 273 appels, 27 M tokens, ~1 h 40 à 5 req/s | 20 s, gratuit |
| Embeddings | 25 M tokens (4 042 appels) | identique |
| Qdrant | 0,53 Go de vecteurs + 0,10 Go de payload | identique |

### Inspecter le découpage d'un document

```bash
python main.py chunks data/test_rapport.txt --blocks
```

Affiche les blocs extraits puis les chunks produits (lignes, section, blocs sources).

---

## Installation

**Prérequis** : Python 3.10+, Docker, une clé API Mistral
([console.mistral.ai](https://console.mistral.ai/)).

```bash
# 1. Environnement virtuel + dépendances
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Clé API
cp .env.example .env
# puis éditez .env pour renseigner MISTRAL_API_KEY

# 3. Base vectorielle
docker compose up -d
```

> **Port Qdrant** : ce projet expose Qdrant sur le port **6343**, pas le 6333
> standard, car une autre instance Qdrant occupe déjà le 6333 sur cette machine.
> Tableau de bord : <http://localhost:6343/dashboard>

### Vérifier l'installation sans consommer de crédits API

```bash
python smoke_test.py                        # parsing -> plan -> chunks (-> Qdrant si dispo)
python -m unittest discover -s tests -t .   # 53 tests, aucune dépendance externe
```

`smoke_test.py` rejoue le pipeline avec un planner déterministe et de faux
embeddings ; l'étage Qdrant est ignoré si le serveur n'est pas démarré.

## Utilisation

### Interface web

```bash
streamlit run app.py
```

Importez un `.txt` dans la colonne de gauche, cliquez sur **Indexer**, posez votre
question. Chaque source affichée indique sa section et ses blocs d'origine.

### Ligne de commande

```bash
python main.py ingest data/                    # indexe tous les .txt du dossier
python main.py ingest ../corpus --recursive    # y compris les sous-dossiers
python main.py chunks data/test_rapport.txt    # inspecte le découpage sans indexer
python main.py ask "Quel est le budget ?"      # pose une question
python main.py status                          # nombre de documents / chunks
python main.py reset                           # vide la base
```

---

## Réglages utiles (`.env`)

| Variable | Défaut | Effet |
|---|---|---|
| `CHUNKING_MODE` | `semantic` | `heuristic` = aucun appel LLM au découpage (rapide et gratuit, utile pour indexer un gros corpus). |
| `PLANNER_PROVIDER` | `mistral` | `ollama` = planification par un modèle local, zéro crédit consommé. |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `localhost:11434` / `llama3.2:3b` | Serveur et modèle local. |
| `OLLAMA_NUM_CTX` | `8192` | Fenêtre de contexte : trop petite, le modèle ne voit pas tous les blocs et invente des `block_id`. |
| `OLLAMA_TIMEOUT` | `180` | Secondes par appel. |
| `TARGET_CHUNK_CHARS` | `1200` | Taille visée d'un chunk. |
| `MAX_CHUNK_CHARS` | `2000` | Plafond garanti par le code, même si le LLM propose plus gros. |
| `MIN_CHUNK_CHARS` | `200` | En deçà, le chunk est fusionné avec le suivant de la même section. |
| `MAX_BLOCK_CHARS` | `1200` | Au-delà, un paragraphe est redécoupé par le parser (sur des phrases). |
| `PLANNER_BATCH_BLOCKS` | `40` | Nombre de blocs envoyés au LLM par appel. |
| `PLANNER_MAX_ATTEMPTS` | `2` | Tentatives avant repli déterministe. |
| `PLANNER_MODEL` | `LLM_MODEL` | Modèle dédié à la planification. |
| `TOP_K` | `4` | Nombre de passages envoyés au LLM. |
| `LLM_MODEL` | `mistral-small-latest` | `mistral-large-latest` pour des réponses de meilleure qualité. |

Deux prompts pilotent le système : celui du **planner** (`rag/planner.py`,
`SYSTEM_PROMPT`) et celui de la **réponse** (`rag/generator.py`).

---

## Détails d'implémentation à connaître

- **Un bloc est atomique** : le parser garantit qu'aucun bloc ne dépasse
  `MAX_BLOCK_CHARS`, donc qu'il tient toujours dans un chunk.
- **Citation par lignes** : `rapport.txt (l. 12-40)`, complétée par le chemin de
  section (`Architecture > Backend`).
- **Identifiants déterministes** : l'ID d'un point Qdrant dérive de `source:index`.
  Réindexer un document supprime d'abord ses anciens chunks : pas de doublon ni de
  point orphelin si le document a raccourci.
- **Similarité cosinus** : le score affiché va de 0 à 1 ; au-delà de ~0,75 le
  passage est en général très pertinent.
- **Réessais automatiques** : les appels API sont retentés 4 fois avec un délai
  exponentiel (`rag/utils.py`), pour absorber les erreurs 429 / réseau.
- **Coût** : le mode `semantic` consomme environ un appel LLM par tranche de 40 blocs.
  Pour ne rien dépenser : `PLANNER_PROVIDER=ollama` (local, plus lent) ou
  `CHUNKING_MODE=heuristic` (déterministe, instantané).

## Pistes pour la suite

Reranking · recherche hybride (dense + BM25) · historique de conversation ·
évaluation (RAGAS) · observabilité · cache d'embeddings.
