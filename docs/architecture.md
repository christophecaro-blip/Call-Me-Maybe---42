# Architecture & approche — call me maybe

> Document de travail personnel (non exigé par le sujet).
> Voir aussi : [concepts.md](concepts.md) pour la théorie, [tasks.md](tasks.md) pour la feuille de route.

---

## 1. Ce qu'on construit, en une phrase

Un programme qui, pour chaque prompt en langage naturel, **laisse le LLM choisir une fonction
et ses arguments**, mais **sans jamais lui laisser la possibilité d'écrire autre chose que du
JSON conforme au schéma** — en filtrant, token par token, ce qu'il a le droit de produire.

```
"What is the sum of 2 and 3?"
        │
        ▼
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

---

## 2. Structure du dépôt

```
CallMeMaybe/
├── Makefile                    # install, run, debug, clean, lint, lint-strict
├── pyproject.toml              # projet uv : python>=3.10, numpy, pydantic, llm_sdk
├── uv.lock                     # généré par uv, À COMMITER
├── README.md                   # exigé (anglais), sections imposées par le sujet
├── .gitignore                  # __pycache__, .venv, .mypy_cache, data/output/ ...
├── .flake8                     # exclure .venv et llm_sdk du lint
├── llm_sdk/                    # copié tel quel depuis attachments_project_ai/ — NE PAS MODIFIER
│   ├── pyproject.toml          #   paquet "llm-sdk" (hatchling) : torch, transformers, huggingface-hub
│   ├── uv.lock
│   └── llm_sdk/__init__.py     #   classe Small_LLM_Model
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── docs/
│   ├── architecture.md         # ce fichier
│   ├── concepts.md
│   └── tasks.md
├── src/
│   ├── __init__.py
│   ├── __main__.py             # point d'entrée : `python -m src` → main() + gestion d'erreurs globale
│   ├── cli.py                  # argparse → CliConfig (pydantic)
│   ├── errors.py               # hiérarchie d'exceptions du projet
│   ├── models.py               # schémas pydantic : FunctionDefinition, ParameterSpec, PromptItem, FunctionCall
│   ├── io_utils.py             # lecture/validation des JSON d'entrée, écriture de la sortie
│   ├── prompting.py            # construction du prompt (chat template Qwen3, description des fonctions)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── vocabulary.py       # chargement du vocab, décodage byte-level, id → str
│   │   └── engine.py           # façade sur Small_LLM_Model : encode, logits → numpy
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── constraints.py      # Constraint (interface) + Choice / Number / Integer / Boolean / String
│   │   └── decoder.py          # boucle générique de décodage contraint (greedy masqué)
│   └── pipeline.py             # FunctionCaller : sélection de fonction → arguments → FunctionCall
└── tests/                      # pytest — utile pour toi, non noté
    ├── test_models.py
    ├── test_constraints.py     # testables SANS le modèle (rapides)
    ├── test_vocabulary.py
    └── test_pipeline.py        # avec le vrai modèle (lents, marqués @pytest.mark.slow)
```

### Pourquoi ce découpage

| Couche | Modules | Dépend du LLM ? | Testable seule ? |
|---|---|---|---|
| Interface utilisateur | `__main__`, `cli` | non | oui |
| Données & I/O | `models`, `io_utils`, `errors` | non | oui |
| Grammaire / contraintes | `decoding/constraints` | **non** | **oui, très facilement** |
| Moteur | `llm/vocabulary`, `llm/engine`, `decoding/decoder` | oui | partiellement |
| Orchestration | `prompting`, `pipeline` | oui | via tests lents |

L'idée clé : **les contraintes ne connaissent pas le modèle**. Elles répondent juste à
« ce morceau de texte peut-il prolonger la valeur en cours ? ». Ça les rend testables en
millisecondes, et c'est là que se trouvent la plupart des bugs subtils.

---

## 3. Flux de données

```
                 ┌──────────────────────┐
 argv ──────────▶│ cli.parse_args()     │──▶ CliConfig (chemins)
                 └──────────────────────┘
                            │
                 ┌──────────▼───────────┐
                 │ io_utils.load_*()    │──▶ list[FunctionDefinition], list[PromptItem]
                 └──────────────────────┘     (JSON invalide / fichier absent → erreur claire)
                            │
                 ┌──────────▼───────────┐
                 │ Engine + Vocabulary  │  chargés UNE fois (lent : modèle + 150k tokens)
                 └──────────────────────┘
                            │
          pour chaque prompt│
                 ┌──────────▼───────────────────────────────────────────────┐
                 │ pipeline.FunctionCaller.call(prompt)                     │
                 │                                                          │
                 │  1. ids = encode(chat_template(system, prompt))          │
                 │  2. force  `{"name": "`                                  │
                 │  3. décode ChoiceConstraint(noms de fonctions)  ◀── LLM  │
                 │  4. force  `", "parameters": {"a": `                     │
                 │  5. décode NumberConstraint                     ◀── LLM  │
                 │  6. force  `, "b": `  ... etc. pour chaque paramètre      │
                 │  7. force  `}}`                                          │
                 │  8. json.loads(texte) → valide via pydantic → FunctionCall│
                 └──────────────────────────────────────────────────────────┘
                            │
                 ┌──────────▼───────────┐
                 │ io_utils.write_out() │──▶ data/output/function_calling_results.json
                 └──────────────────────┘
```

Si un prompt échoue (exception interne inattendue), on log l'erreur sur stderr et on
continue avec les suivants : **une erreur sur un prompt ne doit jamais tuer tout le run**.

---

## 4. L'approche, et les décisions de conception

### D1 — « Remplir les trous » plutôt que « générer du JSON »

On ne demande pas au modèle de produire tout le JSON. La structure (`{`, clés, `:`, `,`, `}`)
est **connue à l'avance** une fois la fonction choisie : c'est le programme qui l'écrit.
Le LLM ne décide que de ce qui est réellement incertain :

1. **le nom de la fonction** (choix fermé parmi N noms),
2. **la valeur de chaque argument** (contrainte par son type).

Formellement c'est toujours du décodage contraint : à chaque position « forcée », un seul
token est autorisé. Avantage majeur : **un token forcé ne nécessite aucun appel au modèle**
(on sait déjà quel token sera choisi), donc gain de vitesse énorme.

> À savoir défendre : « Est-ce que c'est le LLM qui choisit la fonction ? » → Oui : on
> ne fait que masquer les tokens qui ne commencent pas un nom valide ; parmi les candidats
> restants, c'est le logit le plus élevé qui gagne. Aucun mot-clé, aucune regex sur le prompt.

### D2 — Contraintes = petits automates incrémentaux

Chaque type de valeur est un automate qui consomme des caractères :

```python
class Constraint(BaseModel):            # pydantic, cf. exigence du sujet
    def accepts(self, text: str) -> bool: ...    # ce texte peut-il prolonger la valeur ?
    def advance(self, text: str) -> None: ...    # on l'a choisi : mise à jour d'état
    def is_complete(self) -> bool: ...           # la valeur peut-elle s'arrêter ici ?
```

Un token est valide si **tous ses caractères** sont acceptés successivement par l'automate.
C'est le cœur du problème « tokens ≠ caractères » expliqué dans `concepts.md` §5.

### D3 — Terminaison explicite

Chaque valeur se termine par un **marqueur de fin** connu : `"` pour une string, et pour
un nombre/booléen le caractère structurel suivant (`,` ou `}`). Plus une **limite dure**
de tokens par valeur : si elle est atteinte, on ferme la valeur proprement nous-mêmes.
→ La boucle termine toujours, le JSON est toujours valide.

### D4 — Greedy contraint avec recherche paresseuse

Tester 150 000 tokens à chaque pas en Python pur est lent. Stratégie :
trier les logits par ordre décroissant et **tester les candidats un par un jusqu'au premier
valide**. En pratique le bon token est dans les 1–20 premiers. Résultat identique à
« masquer tout puis argmax », pour une fraction du coût. (Optionnel : masques numpy
précalculés pour les états fréquents.)

### D5 — Trois niveaux de garantie

1. **Construction** : le texte est valide par construction (contraintes).
2. **Vérification** : `json.loads` + validation pydantic contre la définition de fonction.
3. **Sérialisation** : la sortie finale est écrite par `json.dump` depuis des objets Python,
   jamais par concaténation de chaînes.

Si (2) échoue malgré (1), c'est un bug de contrainte → message d'erreur clair, pas de crash.

### D6 — Tout ce qui est « classe » est pydantic

Le sujet l'exige. `FunctionDefinition`, `FunctionCall`, `CliConfig`, mais aussi les
contraintes, l'engine, le vocab (avec `model_config = ConfigDict(arbitrary_types_allowed=True)`
pour stocker l'objet `Small_LLM_Model` ou des `np.ndarray`).
Seule exception raisonnable : les exceptions (`class X(Exception)`) — à justifier en soutenance.

### D7 — Pas de valeurs codées en dur

Aucun nom de fonction, aucun exemple du jeu de test dans le code. Les fonctions, leurs
types, leur nombre de paramètres sont lus dans `functions_definition.json`. Le correcteur
**changera** ces fichiers.

---

## 5. Ce que révèlent les fichiers fournis (`attachments_project_ai/`)

### Le SDK (`llm_sdk/llm_sdk/__init__.py`)

| Méthode publique | Comportement réel | Conséquence pour nous |
|---|---|---|
| `Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")` | télécharge (1re fois) et charge tokenizer + modèle ; device auto mps > cuda > cpu | chargement lent → une seule instance ; peut échouer (réseau) → `try/except` |
| `encode(text)` | `torch.Tensor` **2-D** de forme `[1, n]`, `add_special_tokens=False` | convertir : `encode(t)[0].tolist()` → `list[int]` |
| `decode(ids)` | **supprime les tokens spéciaux** | ne pas s'en servir pour construire la table id → texte des tokens spéciaux |
| `get_logits_from_input_ids(ids)` | passage avant **complet** à chaque appel, **pas de KV-cache** ; renvoie `list[float]` du dernier token | coût ∝ longueur du prompt ; `np.asarray(..., dtype=np.float32)` |
| `get_path_to_vocab_file()` | `vocab.json` (token BPE → id) | ne contient **que** les tokens BPE, pas les tokens spéciaux |
| `get_path_to_merges_file()` | `merges.txt` | inutile pour nous |
| `get_path_to_tokenizer_file()` | `tokenizer.json`, avec la liste `added_tokens` | source pour connaître les ids des tokens spéciaux |

Tout ce qui commence par `_` (`_tokenizer`, `_model`, `_device`…) est **interdit** d'usage.

Comme le SDK est un vrai paquet (`pyproject.toml` + hatchling), on le déclare dans **notre**
`pyproject.toml` comme dépendance locale :

```toml
[project]
dependencies = ["numpy", "pydantic", "llm-sdk"]

[tool.uv.sources]
llm-sdk = { path = "llm_sdk", editable = true }
```

→ `uv sync` installe automatiquement torch/transformers **pour le SDK**, sans que notre code
les importe.

### Les données d'entrée

- `functions_definition.json` : une liste de
  `{"name", "description", "parameters": {nom: {"type": ...}}, "returns": {"type": ...}}`.
  Types présents : **`number` et `string` uniquement** — mais d'autres peuvent apparaître en
  correction (`integer`, `boolean`…) : à prévoir.
- `function_calling_tests.json` : liste de `{"prompt": ...}`, 11 prompts, 5 fonctions.
- Cas notables :
  - `fn_substitute_string_with_regex` a **3 paramètres string** et attend une **regex** :
    « Replace all numbers… » → `regex` vaut probablement `\d+`, soit `"\\d+"` en JSON
    → **les échappements de string sont indispensables**, pas optionnels.
  - « …with asterisks » → `replacement` = `*` : le modèle doit **déduire** une valeur absente
    du texte littéral.
  - Le prompt contient des guillemets doubles (`"Hello 34 I'm 233 years old"`) et des apostrophes.

---

## 6. Risques identifiés dès le départ

| Risque | Impact | Parade |
|---|---|---|
| `flake8 .` / `mypy .` analysent `llm_sdk/` et `.venv/` | lint rouge | `exclude` dans `.flake8` et `[tool.mypy]` |
| Tokens « mixtes » (`",` `}}` `"\n`) | JSON cassé si mal gérés | un token est valide ssi **tous** ses caractères passent l'automate |
| Taille des logits ≠ taille du vocab | IndexError / tokens fantômes | masquer les ids sans chaîne associée |
| Mode « thinking » de Qwen3 | le modèle part dans `<think>` | désactiver via le chat template (voir concepts §4) |
| Pas de KV-cache dans le SDK (**confirmé**) | chaque pas recalcule tout le prompt | prompt système court, tokens forcés sans appel modèle |
| `vocab.json` sans les tokens spéciaux | ids spéciaux « inconnus » | les lire dans `tokenizer.json` (`added_tokens`) et les interdire |
| `flake8`/`mypy` sur le code non typé du SDK | lint rouge | exclusion explicite de `llm_sdk/` |
| String qui ne se ferme jamais | boucle infinie | limite de tokens + fermeture forcée |
| Échappements JSON (`\`, `"`) dans les strings (regex du jeu de test !) | JSON invalide | automate de string qui gère `\\`, `\"`, `\n`, `\uXXXX` |
| Type de paramètre inconnu dans les définitions | crash | erreur explicite au chargement (validation pydantic) |
| Modèle non téléchargé / pas de réseau | crash au démarrage | `try/except` autour du chargement + message clair |
