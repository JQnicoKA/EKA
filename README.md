# RAG minimaliste — Mistral + Qdrant

Un pipeline **Retrieval-Augmented Generation** réduit à l'essentiel : on indexe des
des fichiers texte dans une base vectorielle, puis on interroge un LLM en lui fournissant
uniquement les passages pertinents. Chaque réponse est accompagnée de sa source
(document + lignes).

Pas d'observabilité, pas de reranking, pas de stratégie avancée : l'objectif est
de comprendre le mécanisme de bout en bout.

---

## Le pipeline en deux flux

```
INGESTION (une fois par document)
   .txt ──▶ texte ──▶ chunks ──▶ embeddings ──▶ Qdrant
        loader    chunker    MistralEmbedder   vector_store

INTERROGATION (à chaque question)
   question ──▶ embedding ──▶ recherche top-k ──▶ prompt ──▶ LLM ──▶ réponse + sources
             MistralEmbedder    vector_store    generator   Mistral
```

## Architecture du code

| Fichier | Rôle | Classe principale |
|---|---|---|
| `rag/config.py` | Configuration lue depuis `.env` | `Settings` |
| `rag/models.py` | Objets métier échangés entre étages | `Page`, `Chunk`, `RetrievedChunk`, `Answer` |
| `rag/loader.py` | 1. Lecture des fichiers `.txt` | `TextLoader` |
| `rag/chunker.py` | 2. Découpage en morceaux | `TextChunker` |
| `rag/embedder.py` | 3. Vectorisation | `MistralEmbedder` |
| `rag/vector_store.py` | 4. Stockage + recherche | `QdrantVectorStore` |
| `rag/generator.py` | 5. Prompt + appel au LLM | `MistralGenerator` |
| `rag/pipeline.py` | Orchestration des cinq étages | `RAGPipeline` |
| `app.py` | Interface web (Streamlit) | — |
| `main.py` | Interface en ligne de commande | — |
| `smoke_test.py` | Test du pipeline sans appeler l'API Mistral | — |

Chaque étage possède une **classe abstraite** (`DocumentLoader`, `Embedder`,
`VectorStore`, `AnswerGenerator`) et une implémentation concrète. Le pipeline ne
dépend que des abstractions : remplacer Qdrant par autre chose, ou Mistral par un
autre fournisseur, revient à écrire une nouvelle classe sans toucher au reste.

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

> **Déjà fait sur cette machine** : `.venv` (Python 3.11) contient les dépendances,
> `.env` est renseigné, et les deux collections Qdrant sont peuplées. Docker Desktop
> a tendance à s'arrêter entre deux sessions : `open -a Docker` puis attendre le
> démon. S'il reste bloqué, `kill -9` sur `com.docker.backend` puis relancer — un
> `quit` propre ne suffit pas toujours.

> **Port Qdrant** : ce projet expose Qdrant sur le port **6343**, pas le 6333
> standard, car une autre instance Qdrant occupe déjà le 6333 sur cette machine.
> Les deux bases sont donc totalement indépendantes. Tableau de bord :
> <http://localhost:6343/dashboard>

### Vérifier l'installation sans consommer de crédits API

`smoke_test.py` rejoue tout le pipeline (.txt → chunks → Qdrant → recherche →
construction du prompt) en remplaçant Mistral par de faux composants. Utile pour
vérifier que Qdrant répond avant même d'avoir une clé API :

```bash
python smoke_test.py
```

La fixture `tests/fixtures/test_rapport.txt` est versionnée avec le code, pour que
le test tourne sur une machine fraîche.

## Utilisation

### Interface web (recommandé)

```bash
streamlit run app.py
```

Puis, dans le navigateur : importez un `.txt` dans la colonne de gauche, cliquez sur
**Indexer**, et posez votre question.

### Ligne de commande

```bash
python main.py ingest data/               # indexe tous les .txt du dossier
python main.py ingest data/ -r            # ... y compris les sous-dossiers
python main.py ask "Quel est le budget ?" # pose une question
python main.py status                     # nombre de documents / chunks
python main.py reset                      # vide la base
```

---

## Évaluation — EnterpriseRAG-Bench

Le RAG est évalué sur [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) :
un corpus de 511 962 documents d'entreprise simulés et 500 questions avec vérité terrain.

### Où sont les choses

```
Projet_LLM_Base/
├── data/                          le corpus, 511 962 .txt (3,2 Go)
├── EnterpriseRAG-Bench-main/      le harnais d'évaluation
│   ├── questions.jsonl            les 500 questions + gold answers
│   ├── generated_data/sources/    les mêmes documents en .json (requis par le juge)
│   ├── generated_data/uuid_index.json   index dsid -> chemin, pré-construit
│   └── answer_evaluation/         les réponses et les résultats atterrissent ici
└── EKA/
    ├── run_bench.py               génère le fichier de réponses
    └── bench/
        ├── chemins.txt            les 20 722 documents indexés (chemins relatifs à data/)
        └── manifeste.json         722 documents gold + 20 000 distracteurs, graine 0
```

### Les deux collections Qdrant

| Collection | Points | Découpage |
|---|---:|---|
| `documents` | 129 358 | sémantique par blocs, sans recouvrement (branche `semantic-chunking`) |
| `documents_v2` | 144 537 | fenêtre glissante 1000/150 (`main`, actuel) |

Mêmes 20 722 documents de part et d'autre : basculer `QDRANT_COLLECTION` dans `.env`
suffit à comparer les deux stratégies sur le même bench.

### 1. Générer les réponses (Mistral seul)

```bash
python run_bench.py --resume          # 500 questions, ~30 min
python run_bench.py --limit 10        # essai rapide
python run_bench.py --top-k 100       # plus de documents candidats
```

Produit `EnterpriseRAG-Bench-main/answer_evaluation/answers.jsonl`, une ligne par
question : `{"question_id", "answer", "document_ids"}`.

Trois choix à connaître :

- **Réponses en anglais.** Les `gold_answer` sont en anglais ; répondre en français
  ferait chuter la correctness pour une raison étrangère au RAG. Le script passe un
  `system_prompt` anglais à `MistralGenerator`. `--francais` rétablit le comportement normal.
- **`--top-k 20` chunks → `--max-docs 10` documents.** Le benchmark raisonne en
  documents : les chunks sont dédupliqués par document en gardant le meilleur rang.
- **`document_ids`** est extrait du nom de fichier : `dsid_ae06...__multipart.txt` → `dsid_ae06...`.

### 2. Lancer le juge (Anthropic ou OpenAI)

Le juge du benchmark n'accepte **que** `openai` ou `anthropic` (`src/llm/factory.py`),
et lit `LLM_API_KEY` — pas `ANTHROPIC_API_KEY`. Il faut donc mapper :

```bash
cd ../EnterpriseRAG-Bench-main
LLM_PROVIDER=anthropic LLM_API_KEY="$ANTHROPIC_API_KEY" \
  python -m src.scripts.answer_evaluation.metrics_based_eval \
    --answers-file answer_evaluation/answers.jsonl \
    --parallelism 4 --resume
```

Résultats dans `answer_evaluation/results.json`. Les quatre métriques :
correctness, completeness, document recall, invalid extra documents.

### Résultats mesurés (14 septembre 2026, `documents_v2`, `top_k=20`)

500 réponses générées en 30,7 min, zéro échec. **Document recall brut : 69,3 %**
— calculé sans le juge, donc gratuit, avec la formule exacte du dépôt
(`metrics_based_eval.py:438` : moyenne de `|gold ∩ proposés| / |gold|` sur les
470 questions à gold non vide).

| Catégorie | n | Recall | Gold complet | Rien trouvé |
|---|---:|---:|---:|---:|
| `constrained` | 30 | 95,0 % | 90,0 % | 0 % |
| `intra_document_reasoning` | 40 | 82,5 % | 82,5 % | 17,5 % |
| `conflicting_info` | 20 | 80,0 % | 70,0 % | 10,0 % |
| `miscellaneous` | 20 | 80,0 % | 80,0 % | 20,0 % |
| `project_related` | 40 | 75,4 % | 40,0 % | 0 % |
| `basic` | 175 | 71,4 % | 71,4 % | 28,6 % |
| `semantic` | 125 | **54,4 %** | 54,4 % | **45,6 %** |
| `completeness` | 20 | **44,3 %** | 15,0 % | 15,0 % |
| `info_not_found` / `high_level` | 30 | — | pas de gold | |

Lecture :

- **`semantic` est le point faible** — 125 questions, une sur deux ne remonte rien.
  C'est la catégorie conçue sans recouvrement de mots-clés : un embedding seul, sans
  reranking ni recherche hybride, y est en difficulté par construction.
- **`completeness` et `project_related` sont bridés par le plafond**, pas forcément
  par la recherche : ces questions attendent jusqu'à 10 documents, or 20 chunks à
  ~11 chunks/document ne couvrent que 2 ou 3 documents distincts. À retester avec
  `--top-k 100`.
- **~9 documents en trop par question**, conséquence directe de `--max-docs 10` face
  à des gold sets d'un seul document.

### ⚠️ Ces scores ne sont PAS comparables au leaderboard

**L'index ne contient que 20 722 documents sur 511 962, soit 4 %** — les 722 documents
gold plus 20 000 distracteurs tirés au sort (voir `bench/manifeste.json`). Le champ de
recherche est **25 fois plus petit** que celui du leaderboard. Le 69,3 % est donc
structurellement gonflé, et le juge n'y changerait rien : correctness et completeness
sont tout autant concernées, puisqu'une réponse juste suppose d'abord le bon document.

Pour un chiffre comparable, il faudrait indexer le corpus entier :

```
~756 M tokens d'embeddings  (mesuré : 1 477 tokens/document)
~20 heures                  (mesuré : 48 min pour 20 722 documents)
```

**En l'état, les scores valent en relatif, pas en absolu.** C'est suffisant — et
gratuit — pour répondre aux vraies questions : le chunking sémantique bat-il la
fenêtre glissante ? `top_k=100` débloque-t-il `completeness` ? Le préfixe de chemin
aide-t-il les questions `semantic` ? Le recall brut se recalcule sans le moindre appel
au juge, c'est la métrique la plus discriminante et la moins chère.

Le juge devient utile pour savoir si les réponses sont *justes*, pas seulement si les
bons documents remontent — donc pour évaluer le couple retrieval + génération.

### Limites du compte Mistral

| Modèle | État (septembre 2026) |
|---|---|
| `mistral-embed` | ✅ fonctionne |
| `open-mistral-7b` | ✅ fonctionne — **le seul modèle de chat disponible** |
| `mistral-small-latest` | ❌ 429 Rate limit, de façon persistante |
| `mistral-large-latest` | ❌ 403 « not available in your subscription tier » |

La correctness mesurée sera donc celle d'un modèle 7B, pas celle du RAG dans l'absolu.
Le retrieval, lui, ne dépend que de `mistral-embed` et reste valable quel que soit le
générateur.

---

## Les trois dossiers de données

Ne pas les confondre — ils ont des rôles et des durées de vie très différents :

| Dossier | Contenu | Statut |
|---|---|---|
| `Projet_LLM_Base/data/` | le **corpus du benchmark**, 511 962 `.txt` (3,2 Go) | ne jamais y écrire ; `bench/chemins.txt` y pointe |
| `Projet_LLM_Base/data_perso/` | vos **documents importés** via l'interface (`DATA_DIR`) | hors du dépôt git, donc jamais commité |
| `EKA/tests/fixtures/` | la fixture de `smoke_test.py` | versionnée avec le code |

Le dossier d'import est délibérément **hors du dépôt** : un document déposé par
l'utilisateur ne doit pas pouvoir se retrouver dans un commit. Il est aussi un
*frère* du corpus et non un fils, pour qu'une réécriture du corpus (déjà arrivée
une fois) ou une ingestion récursive ne l'emporte pas au passage.

L'import n'accepte que le `.txt`, vérifié à trois niveaux : le `file_uploader`
côté navigateur, un contrôle d'extension avant écriture sur disque (`app.py`), et
`TextLoader.supports()` au chargement.

---

## Réglages utiles (`.env`)

| Variable | Défaut | Effet |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Taille d'un chunk en caractères. Plus petit = recherche plus précise, mais contexte plus fragmenté. |
| `CHUNK_OVERLAP` | `150` | Recouvrement entre deux chunks. Évite de couper une idée en deux. |
| `TOP_K` | `4` | Nombre de passages envoyés au LLM. Plus élevé = plus de contexte, mais plus de bruit et de coût. |
| `LLM_MODEL` | `mistral-small-latest` | ⚠️ **inutilisable sur ce compte** (429 permanent). Mettre `open-mistral-7b`, seul modèle de chat qui réponde. Voir « Limites du compte Mistral ». |
| `QDRANT_COLLECTION` | `documents` | `documents_v2` pour l'index en fenêtre glissante. |

Le prompt système (la consigne donnée au LLM) se trouve dans
`rag/generator.py` — c'est le premier levier à ajuster pour améliorer les réponses.

---

## Détails d'implémentation à connaître

- **Références en numéros de ligne** : un fichier texte n'a pas de pages. Chaque
  chunk retient les lignes d'où il provient, d'où des citations du type
  `notes.txt (l. 12-40)`. Le contenu est lu **sans nettoyage**, faute de quoi les
  numéros ne correspondraient plus au fichier ouvert dans un éditeur.
- **Identifiants déterministes** : l'ID d'un point Qdrant est dérivé de
  `source:index`. Réindexer le même fichier met à jour les points au lieu de
  créer des doublons — mais s'il a raccourci, les points en trop subsistent.
- **Similarité cosinus** : le score affiché va de 0 à 1 ; au-delà de ~0,75 le
  passage est en général très pertinent.
- **Format unique** : seul le `.txt` est accepté (`TextLoader.SUFFIXES`). Ajouter
  le `.md` tient en une entrée de ce tuple ; un autre format demande une nouvelle
  classe implémentant `DocumentLoader`.
- **Réessais automatiques** : 5 tentatives avec délais 5, 10, 20 puis 40 s, soit
  75 s cumulés (`rag/utils.py`). C'est calibré pour franchir la réinitialisation d'un
  quota par minute : un backoff plus court abandonnait avant la fin de la fenêtre et
  faisait perdre le lot — donc payer sa vectorisation pour rien.
- **Ingestion en lots** : `ingest_directory` accumule les chunks **entre documents**
  avant de vectoriser (`batch_chunks=256`). Sans ce tampon, un document de 7 chunks
  déclenchait un appel réseau à lui seul : mesuré, cela divise par 4,2 le nombre de
  requêtes. `resume=True` saute les documents déjà en base ; le tampon n'étant vidé
  qu'entre deux documents, un document présent en base y est toujours intégralement.
- **Arrêt si la base tombe** : un échec d'écriture lève une `RuntimeError` et
  interrompt l'ingestion. Continuer reviendrait à payer la vectorisation de tout le
  reste du corpus pour la jeter — c'est arrivé, ça a coûté ~3,5 M de tokens.

## Pistes pour la suite

Par ordre de rendement attendu, d'après les mesures ci-dessus :

1. **`--top-k 100`** — vérifier si `completeness` (15 % de gold complet) et
   `project_related` (40 %) sont bridés par le plafond de documents ou par la
   recherche. Coût : 30 min de Mistral, aucun juge.
2. **Reranking ou recherche hybride (dense + BM25)** — c'est la réponse classique au
   point faible `semantic` (54,4 %), où l'embedding seul ne suffit pas.
3. **Comparer `documents` et `documents_v2`** sur le même bench pour trancher enfin la
   question du chunking sémantique.
4. **Préfixer les chunks avec leur chemin** puis réembedder (~30 M de tokens) — mesuré
   à +0,06 de similarité sur les questions évoquant l'origine du document, +0,01 sinon.
   Le champ `chemin` est déjà dans le payload Qdrant, donc déjà utilisable en filtrage.

Plus loin : historique de conversation · observabilité · cache d'embeddings.
