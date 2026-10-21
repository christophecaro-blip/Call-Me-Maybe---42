# Concepts — call me maybe

> Tout ce qu'il faut comprendre pour **concevoir** le projet et pour le **défendre**.
> Chaque section se termine par « ✅ Tu dois savoir expliquer » : c'est ta checklist de soutenance.

Sommaire
1. [Function calling](#1-function-calling)
2. [Comment un LLM génère du texte](#2-comment-un-llm-génère-du-texte)
3. [Tokenisation : BPE byte-level (Qwen)](#3-tokenisation--bpe-byte-level-qwen)
4. [Le prompt : chat template Qwen3](#4-le-prompt--chat-template-qwen3)
5. [Décodage contraint](#5-décodage-contraint)
6. [Génération pilotée par le schéma](#6-génération-pilotée-par-le-schéma)
7. [Les grammaires des valeurs JSON](#7-les-grammaires-des-valeurs-json)
8. [Terminaison et cas limites](#8-terminaison-et-cas-limites)
9. [Performance](#9-performance)
10. [Validation et robustesse](#10-validation-et-robustesse)
11. [Outillage Python imposé](#11-outillage-python-imposé)
12. [Glossaire](#12-glossaire)

---

## 1. Function calling

Un LLM « classique » répond en prose : *"The sum of 40 and 2 is 42."*
Un système de function calling **ne répond pas** : il produit une **instruction exécutable**
par un programme :

```json
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
```

Deux sous-problèmes très différents :

| Sous-problème | Nature | Qui le résout ici |
|---|---|---|
| **Quoi** appeler, avec **quelles valeurs** | compréhension du langage | le LLM |
| **Sous quelle forme** (JSON, types, clés) | syntaxe stricte | le programme (contraintes) |

Toute la philosophie du projet tient dans ce tableau : on laisse au modèle ce qu'il sait
faire (comprendre), on lui retire ce qu'il fait mal (respecter une syntaxe à 100 %).

**Usages réels** : agents qui appellent des API, extraction de données structurées depuis du
texte, assistants qui pilotent des outils.

✅ Tu dois savoir expliquer
- la différence entre « répondre » et « produire un appel de fonction » ;
- pourquoi un petit modèle (0.6B) échoue souvent à produire du JSON par simple prompting.

---

## 2. Comment un LLM génère du texte

### Génération autorégressive

Un LLM est une fonction : `séquence de tokens → un score pour chaque token possible suivant`.

```
ids = encode(prompt)
répéter :
    logits = model(ids)            # vecteur de taille |V| ≈ 151 000
    next_id = choisir(logits)
    ids.append(next_id)
jusqu'à condition d'arrêt
```

Chaque token produit est **réinjecté** dans l'entrée : le modèle ne « voit » que la séquence,
il n'a pas de plan global. C'est pour ça qu'on peut l'orienter pas à pas.

### Logits, softmax, probabilités

- **Logit** = score brut, non borné (ex : `12.3`, `-4.1`).
- **Softmax** les transforme en probabilités : `p_i = exp(l_i) / Σ_j exp(l_j)`.
- Propriété cruciale : si `l_i = -∞` alors `exp(l_i) = 0` → **probabilité nulle**.
  C'est exactement le levier du décodage contraint.

### Stratégies de choix

| Stratégie | Principe | Ici |
|---|---|---|
| **Greedy** | `argmax(logits)` | ✅ déterministe, reproductible, idéal pour de l'extraction |
| Sampling | tirage selon `p` (avec température) | inutile, ajoute du bruit |
| Beam search | garde les k meilleures séquences | possible en bonus, coûteux |

Note : greedy ne nécessite même pas le softmax (argmax des logits = argmax des probas).
Le softmax/log-softmax devient utile si tu veux **comparer des scores de séquences**
(ex : somme des log-probas de chaque nom de fonction).

✅ Tu dois savoir expliquer
- ce que retourne `get_logits_from_input_ids` et pourquoi sa taille ≈ taille du vocabulaire ;
- pourquoi mettre un logit à `-inf` revient à interdire un token ;
- pourquoi greedy est un bon choix ici.

---

## 3. Tokenisation : BPE byte-level (Qwen)

### Un token n'est ni un caractère ni un mot

Qwen utilise un **BPE byte-level** (même famille que GPT-2). Le vocabulaire contient des
morceaux de texte de longueur variable :

```
"What is the sum of 265 and 345?"
→ ["What", "Ġis", "Ġthe", "Ġsum", "Ġof", "Ġ", "2", "6", "5", "Ġand", "Ġ", "3", "4", "5", "?"]
```

(découpage illustratif — **vérifie-le toi-même** avec le SDK, c'est le premier exercice de `tasks.md`).

Points à observer :
- **`Ġ` = espace** devant le mot, **`Ċ` = retour à la ligne** (`\n`). Ce ne sont pas les vrais
  caractères : c'est l'encodage « bytes → unicode imprimable » de GPT-2.
- Qwen **découpe les nombres chiffre par chiffre** → très pratique pour contraindre les nombres.
- Il existe des tokens « mixtes » qui mélangent ce que tu considères comme des catégories
  différentes : `",`, `"}`, `}}`, `":`, `Ġ"`, `.Ċ`… **c'est LE piège du projet**.

### Le fichier de vocabulaire

`get_path_to_vocab_file()` télécharge et renvoie le chemin de **`vocab.json`** :
`{"token_string": id, ...}`, avec les tokens BPE **uniquement**. Les tokens spéciaux
(`<|im_start|>`, `<think>`…) n'y sont pas : ils figurent dans la clé `added_tokens` de
`tokenizer.json` (`get_path_to_tokenizer_file()`).

Pour le décodage contraint il faut l'inverse : `id → texte réel`. Deux approches :

1. **Inverser le mapping « bytes → unicode » de GPT-2** : chaque caractère du token correspond à
   un octet ; on reconstitue les octets puis on décode en UTF-8.
   ```python
   def bytes_to_unicode() -> dict[int, str]:
       bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
            + list(range(ord("®"), ord("ÿ") + 1))
       cs = bs[:]
       n = 0
       for b in range(256):
           if b not in bs:
               bs.append(b)
               cs.append(256 + n)
               n += 1
       return dict(zip(bs, map(chr, cs)))
   ```
   Inverse ce dict (`char → byte`), puis pour un token : `bytes(byte_of[c] for c in token)`.
2. **Utiliser `decode([id])` du SDK** pour chaque id, une fois au démarrage, et mettre en cache.
   Plus simple, plus lent au démarrage (≈150k appels) — mesure-le. Attention : le `decode`
   du SDK passe `skip_special_tokens=True`, donc un token spécial donne `""` ; et un fragment
   UTF-8 isolé donne `"�"`. Les deux doivent être interdits.

L'approche 1 est plus intéressante à défendre : elle montre que tu as compris l'encodage.

⚠️ Certains tokens sont des **fragments d'un caractère UTF-8 multi-octets** (accents, emojis) :
leurs octets seuls ne forment pas un UTF-8 valide. Option simple et défendable : les exclure
(ou les décoder avec `errors="replace"` et les interdire dans les contraintes). Ça ne gêne
pas les tests en anglais, mais **sache le dire** si on te demande « et les caractères spéciaux ? ».

### Tokens spéciaux

`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>`, `</think>`… Ils ont des ids réservés.
Ils doivent **toujours** être interdits pendant la génération des valeurs.

### Taille des logits ≠ taille du vocab

La couche de sortie du modèle est arrondie (attendu pour Qwen3 : 151 936 logits) alors que
`vocab.json` contient ~151 643 tokens, plus ~26 tokens spéciaux. Les ids sans texte associé
doivent être masqués.
→ **Vérifie les trois tailles** (logits, `vocab.json`, `added_tokens`) en Phase 1.

✅ Tu dois savoir expliquer
- pourquoi on ne peut pas raisonner « caractère par caractère » sur la sortie du modèle ;
- ce que signifient `Ġ` et `Ċ` ;
- comment tu passes d'un id à la chaîne réelle ;
- pourquoi un token comme `",` pose problème et comment ton code le gère.

---

## 4. Le prompt : chat template Qwen3

Qwen3 est un modèle **instruct** entraîné sur un format de conversation précis :

```
<|im_start|>system
{instructions + liste des fonctions}<|im_end|>
<|im_start|>user
{prompt utilisateur}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

- Respecter ce format améliore nettement la qualité des choix du modèle.
- **Qwen3 a un mode « thinking »** : par défaut il commence par `<think>` et raisonne longuement.
  Ajouter un bloc `<think>\n\n</think>\n\n` **vide** à la fin du prompt assistant le désactive
  (c'est ce que fait `enable_thinking=False` dans le template officiel). Sinon : temps perdu.
- Le décodage contraint **ne remplace pas** un bon prompt : il garantit la *forme*, le prompt
  aide le *fond* (bonne fonction, bonnes valeurs). Les deux sont complémentaires.

Contenu utile du message system (court, chaque token coûte — cf. §9) :
- une phrase de rôle (« You translate requests into function calls ») ;
- la liste des fonctions : nom, description, paramètres typés ;
- éventuellement **un ou deux exemples génériques** (few-shot) qui ne sont pas tirés du jeu
  de test. Très efficace sur un 0.6B, à mesurer.

> Ce qui est interdit (sujet §5.3.3) : compter sur le prompt **seul** pour obtenir du JSON.
> Ce qui est autorisé : utiliser le prompt pour **aider** le modèle, la forme étant garantie
> par les contraintes.

✅ Tu dois savoir expliquer
- pourquoi utiliser le chat template ;
- ce qu'est le mode thinking et comment tu le neutralises ;
- la répartition des rôles prompt vs contraintes.

---

## 5. Décodage contraint

### Principe

À chaque pas :
1. le modèle produit les logits ;
2. on détermine les tokens **qui gardent la sortie valide** (syntaxe JSON **et** schéma) ;
3. on met les autres à `-inf` ;
4. on choisit (argmax) parmi les restants.

Garantie : **par récurrence**, si le préfixe est valide et qu'on n'ajoute que des tokens qui
gardent un préfixe valide, et qu'on ne s'arrête que dans un état final, la sortie est valide.
Il faut juste que l'ensemble autorisé ne soit **jamais vide** (sinon impasse — voir §8).

### Le problème central : tokens vs caractères

Les règles JSON s'expriment en **caractères**, le modèle produit des **tokens multi-caractères**.
Solution générale : décrire la grammaire comme un **automate à états** qui consomme des
caractères, et dire :

> un token est valide ⇔ l'automate peut consommer **tous** ses caractères, l'un après l'autre,
> sans tomber dans un état d'erreur.

```
état courant : dans un nombre, déjà lu "26"
token "5"   → '5' ok                         → valide
token "5."  → '5' ok, '.' ok                 → valide
token "5,"  → '5' ok, ',' = marqueur de fin  → valide si la fin est autorisée ici (voir §8)
token "Ġ5"  → ' ' refusé                     → invalide
token "abc" → 'a' refusé                     → invalide
```

### Automate fini (FSM) en bref

Un ensemble d'**états**, un état **initial**, des états **finaux** (acceptants), et une
**fonction de transition** `(état, caractère) → nouvel état | erreur`. Les nombres JSON, les
booléens, les strings sont des langages réguliers → reconnaissables par FSM. Tu peux coder
l'état sous forme d'enum + quelques champs (pydantic).

### Le trie (arbre de préfixes) pour les choix fermés

Pour choisir parmi `["fn_add_numbers", "fn_greet", "fn_reverse_string"]` :
un token est valide si `déjà_généré + token` est **un préfixe** d'au moins un nom (ou un nom
complet suivi du `"` fermant). Un trie ou simplement `any(name.startswith(candidate))` suffit
(N est petit).

Variante plus « propre » à mentionner : **scorer chaque nom complet** par la somme des
log-probabilités de ses tokens et prendre le max. Plus robuste quand deux noms partagent un
long préfixe, mais N fois plus d'appels modèle. Greedy sur trie = bon compromis.

✅ Tu dois savoir expliquer
- les 4 étapes du décodage contraint et pourquoi la validité est **garantie** ;
- comment tu testes si un token multi-caractères est autorisé ;
- en quoi le choix de la fonction est fait par le LLM et pas par une heuristique.

---

## 6. Génération pilotée par le schéma

Une fois la fonction connue, le **squelette** de la sortie est entièrement déterminé :

```
{"name": "fn_add_numbers", "parameters": {"a": ⟨number⟩, "b": ⟨number⟩}}
 └──── forcé ────┘└choix┘└────── forcé ──────┘└ LLM ┘└forcé┘└ LLM ┘└forcé┘
```

- **Segments forcés** : on encode le texte fixe et on ajoute directement ses ids à la séquence.
  Pas d'appel au modèle → rapide. (Subtilité : l'encodage d'un segment isolé peut différer de
  l'encodage « naturel » du texte complet. Sans conséquence pour la validité ; à mentionner
  si on te pose la question.)
- **Trous** : décodage contraint par le type du paramètre (lu dans `functions_definition.json`).

Ordre des paramètres : celui de la définition (les dict Python préservent l'ordre d'insertion,
pydantic aussi).

Le champ `"prompt"` de la sortie **n'est pas généré** : c'est l'entrée recopiée par le programme.

✅ Tu dois savoir expliquer
- quelles parties de la sortie sont décidées par le modèle et lesquelles par le code ;
- pourquoi ça reste du décodage contraint (un seul token autorisé = forcé).

---

## 7. Les grammaires des valeurs JSON

### Nombre (`number`)

Grammaire JSON officielle : `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`

```
         '-'          '0'
 START ──────▶ SIGN ──────▶ ZERO* ──'.'──▶ DOT ──digit──▶ FRAC* ──e/E──▶ EXP ...
   │             │ 1-9                       ▲
   │'0'          ▼                           │
   └────────▶ INT* ◀── digit ──┘ ───'.'──────┘
                                   (* = état acceptant)
```

Simplification acceptable : ignorer l'exposant (à documenter). **Limiter la longueur**
(ex. 20 caractères) pour éviter les nombres infinis.

Sortie : pour un paramètre `number`, l'exemple du sujet montre `2.0` → convertis en `float`
avant écriture. Pour `integer` (si présent) : pas de `.`, convertis en `int`.

### Booléen (`boolean`)

Choix fermé `["true", "false"]` → réutilise exactement la contrainte de choix (trie).

### Chaîne (`string`)

```
 ouvrant " forcé ─▶ CONTENT ──'"'──▶ FIN
                     │  ▲
                 '\\'│  │ " \\ / b f n r t  ou  uXXXX
                     ▼  │
                    ESCAPE
```

- Interdits dans CONTENT : `"` (sauf comme fin), caractères de contrôle `< 0x20`.
- `\` ouvre un échappement : seuls `" \ / b f n r t u` sont légaux ensuite.
- Le cas difficile : **regex** en argument — il est **dans le jeu de test fourni**
  (`fn_substitute_string_with_regex`, « Replace all numbers in … »). Le modèle doit écrire
  `\\d+` en JSON pour obtenir `\d+` après `json.loads`. Ton automate l'y oblige.
  Version 1 (pour avancer) : interdire `\` ; version 2 (**nécessaire** pour ce prompt) :
  gérer les échappements.
- Autre piège : le modèle a appris que `\d` « existe » ; s'il veut écrire `\d` (invalide en
  JSON), l'automate refuse le `d` après `\` → il devra choisir `\\`. C'est le décodage contraint
  qui corrige le modèle, pas le prompt.
- Les **chaînes vides** (`""`) sont valides : le `"` fermant doit être autorisé dès le départ.

### Autres types éventuels

Le correcteur peut fournir d'autres définitions. Si un type est inconnu (`array`, `object`…),
décide d'une politique claire : soit l'implémenter, soit **refuser proprement au chargement**
avec un message explicite. Jamais de crash.

✅ Tu dois savoir expliquer
- l'automate de chaque type, et dessiner celui des nombres au tableau ;
- comment tu gères les échappements et la chaîne vide ;
- pourquoi `2` devient `2.0` pour un `number`.

---

## 8. Terminaison et cas limites

| Situation | Problème | Solution |
|---|---|---|
| Fin d'un nombre | rien dans le nombre ne dit « stop » | la fin = le caractère structurel suivant (`,` ou `}`), autorisé **seulement si** l'automate est dans un état acceptant |
| Token `5,` / `"}` / `}}` | dépasse la valeur | soit l'interdire (ne garder que les tokens qui finissent pile sur le marqueur), soit accepter et **tenir compte** des caractères en trop dans le segment forcé suivant. La 1re option est bien plus simple |
| String qui ne se ferme pas | boucle infinie | `max_tokens` par valeur → on ferme nous-mêmes (`"`) |
| Ensemble autorisé vide | impasse | ne devrait pas arriver si chaque état non final a au moins une sortie ; défense : fermeture forcée + log |
| Nombre `-` seul ou `1.` à la limite | invalide | à la fermeture forcée, compléter (`-0`, `1.0`) ou recommencer |

Règle d'or : **à aucun moment** ton programme ne doit pouvoir écrire un JSON invalide, même
dans les chemins d'erreur.

✅ Tu dois savoir expliquer
- comment ta génération se termine, dans tous les cas ;
- ce qui se passe si le modèle « veut » écrire n'importe quoi indéfiniment.

---

## 9. Performance

Contrainte : **< 5 minutes pour tout le fichier de tests**.

### D'où vient le coût

- `get_logits_from_input_ids(ids)` est **sans KV-cache** (confirmé dans le code du SDK :
  `self._model(input_ids=...)` sur toute la séquence à chaque appel). Coût d'un pas ∝ longueur
  du prompt. → Prompt système **court**.
- Il renvoie une `list[float]` de ~152k éléments (conversion tensor → liste à chaque appel) :
  la transformer **une fois** en `np.ndarray` et ne travailler qu'en numpy ensuite.
- Tester la validité de ~150k tokens en Python à chaque pas : très lent si fait naïvement.

### Leviers (du plus rentable au plus avancé)

1. **Tokens forcés sans appel modèle** : la moitié de la sortie environ.
2. **Recherche paresseuse** : `order = np.argsort(-logits)` puis tester les candidats dans cet
   ordre, s'arrêter au premier valide. Même résultat que le masque complet + argmax.
   (`np.argpartition` pour un top-k si tu veux encore gratter.)
3. **Précalculs au démarrage** : texte décodé de chaque token, ensembles « tokens 100 % chiffres »,
   « tokens sans `"` ni `\` ni contrôle »… → masques `numpy` booléens réutilisables.
4. Charger modèle et vocab **une seule fois** pour tous les prompts.
5. **Mesurer** (`time.perf_counter`) : temps de chargement, temps par prompt, nombre d'appels
   modèle. Ce sont les chiffres de la section « Performance analysis » du README.

✅ Tu dois savoir expliquer
- pourquoi un long prompt système ralentit tout ;
- pourquoi la recherche paresseuse donne le même résultat que le masquage complet ;
- tes mesures de temps.

---

## 10. Validation et robustesse

### pydantic (v2)

```python
class ParameterSpec(BaseModel):
    type: Literal["number", "integer", "string", "boolean"]

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec
```

- `Model.model_validate(data)` valide un dict ; `TypeAdapter(list[Model]).validate_python(...)`
  pour une liste.
- `ValidationError` donne des messages précis → transforme-les en erreur utilisateur lisible.
- Ces modèles correspondent au vrai `functions_definition.json` fourni. Il ne contient que
  `number` et `string`, mais le `Literal` doit lister tous les types que tu sais gérer :
  tout autre type est rejeté proprement au chargement.

### Erreurs à gérer (sujet : « never crash »)

| Cas | Comportement attendu |
|---|---|
| fichier d'entrée absent | message clair, code de sortie ≠ 0 |
| JSON invalide | message avec ligne/colonne (`json.JSONDecodeError` les fournit) |
| JSON valide mais mauvais schéma | message pydantic résumé |
| liste vide de fonctions / de prompts | message clair (pas de fonction → impossible d'appeler) |
| dossier de sortie inexistant | le créer (`Path.mkdir(parents=True, exist_ok=True)`) |
| sortie non inscriptible | message clair |
| échec de chargement du modèle | message clair |
| `Ctrl+C` | sortie propre |

Pattern : exceptions métier dans `errors.py` (`InputFileError`, `ModelLoadError`…), un seul
`try/except` global dans `__main__.py` qui les convertit en message + `sys.exit(1)`, et un
`try/except` **par prompt** dans le pipeline pour ne pas perdre tout le lot.

Fichiers : toujours `with open(...)` (context manager), encodage explicite `utf-8`.

✅ Tu dois savoir expliquer
- comment ton programme réagit à chaque cas du tableau (et le montrer en live) ;
- pourquoi les erreurs sont converties à un seul endroit.

---

## 11. Outillage Python imposé

### uv

```bash
uv init --python 3.10        # ou version plus récente
uv add numpy pydantic
uv sync                      # ce que lance le correcteur : doit TOUT installer
uv run python -m src
```

`llm_sdk/` est copié à côté de `src/`. Il a ses propres dépendances (torch, transformers,
huggingface-hub — autorisées **à l'intérieur du SDK**, interdites dans ton code).
Le SDK fourni est un vrai paquet (`llm_sdk/pyproject.toml`, nom `llm-sdk`, code dans
`llm_sdk/llm_sdk/`) → on le déclare comme **dépendance locale** :

```toml
[project]
dependencies = ["numpy", "pydantic", "llm-sdk"]

[tool.uv.sources]
llm-sdk = { path = "llm_sdk", editable = true }
```

`uv sync` installe alors le SDK **et** ses dépendances. Dans ton code : `from llm_sdk import Small_LLM_Model`.

Subtilité : à la racine, le dossier externe `llm_sdk/` (sans `__init__.py`) pourrait être pris
pour un *namespace package*. Python préfère toujours un paquet « régulier » (avec
`__init__.py`, ici celui installé par uv) à un namespace package, donc l'import fonctionne.
Vérifie-le quand même avec `uv run python -c "import llm_sdk; print(llm_sdk.__file__)"`.

Le test décisif : `git clone` dans un dossier vierge → `uv sync` → `uv run python -m src`.

### mypy — signification des flags du `make lint`

| Flag | Effet |
|---|---|
| `--disallow-untyped-defs` | toute fonction doit être annotée |
| `--check-untyped-defs` | vérifie aussi le corps des fonctions non annotées |
| `--warn-return-any` | signale un `Any` retourné là où un type précis est annoncé |
| `--warn-unused-ignores` | signale les `# type: ignore` inutiles |
| `--ignore-missing-imports` | ne bloque pas sur les libs sans stubs |

`--strict` (optionnel) ajoute notamment l'interdiction des génériques nus (`list` au lieu de
`list[int]`). Vise-le dès le début, c'est moins cher que de corriger à la fin.

### flake8

PEP 8 + erreurs simples. Longueur de ligne 79 par défaut. Exclure `.venv`, `llm_sdk`.

### Makefile

`install`, `run`, `debug` (`python -m pdb -m src`), `clean`, `lint`, `lint-strict`.

✅ Tu dois savoir expliquer
- ce que fait `uv sync` et pourquoi `uv.lock` est commité ;
- chaque flag mypy de la règle `lint`.

---

## 12. Glossaire

| Terme | Définition |
|---|---|
| **Token** | unité de texte du vocabulaire du modèle (morceau de mot, ponctuation, chiffre…) |
| **Vocabulaire** | ensemble de tous les tokens, chacun avec un id entier |
| **BPE** | Byte-Pair Encoding : construit les tokens en fusionnant les paires les plus fréquentes |
| **Logit** | score brut du modèle pour un token candidat |
| **Softmax** | transforme les logits en probabilités |
| **Greedy** | on prend toujours le token de score max |
| **Décodage contraint** | masquer les tokens interdits avant le choix |
| **Masque** | vecteur booléen « autorisé / interdit » sur tout le vocabulaire |
| **FSM / automate** | machine à états qui reconnaît un langage caractère par caractère |
| **Trie** | arbre de préfixes, pour tester rapidement « est-ce le début d'un mot connu ? » |
| **Token forcé** | position où un seul token est autorisé : ajouté sans appel au modèle |
| **Chat template** | format de conversation attendu par un modèle instruct |
| **Few-shot** | exemples donnés dans le prompt pour guider le modèle |
| **KV-cache** | mémorisation des calculs des tokens précédents (absent ici a priori) |
| **Schéma** | description des clés et types attendus (ici : `functions_definition.json`) |
