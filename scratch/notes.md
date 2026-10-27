# Notes — Phase 1 : exploration du SDK et du tokenizer

> Suit l'ordre des étapes de `docs/tasks.md`, phase 1.
> Script associé indiqué à chaque section. Les `⏳ À compléter` sont les trous restants.

## Sommaire

1. [Chargement du modèle](#1-chargement-du-modèle) — `01_load.py`
2. [vocab.json, tokenizer.json et logits](#2-vocabjson-tokenizerjson-et-logits) — `03_vocab.py`
3. [Tokens mixtes](#3-tokens-mixtes) — `03_vocab.py`
4. [Exercice : le JSON reste-t-il valide ?](#4-exercice--le-json-reste-t-il-valide-) — `04_json_check.py`
5. [Token spécial `<|im_start|>`](#5-token-spécial-im_start) — `05_special_tokens.py`
6. [Découpage : `Ġ` et chiffres](#6-découpage--ġ-et-chiffres) — `06_tokens.py`
7. [Génération non contrainte](#7-génération-non-contrainte) — `02_generate.py`
8. [Temps d'un pas](#8-temps-dun-pas) — `02_generate.py`
9. [Récapitulatif « Terminé quand… »](#9-récapitulatif--terminé-quand-)

---

## 1. Chargement du modèle

- **Temps de chargement : ~2,1 s** (modèle déjà dans le cache Hugging Face, CPU).
  Coût payé **une seule fois** par exécution : le programme final n'instancie le modèle qu'une fois.
- **Matériel réellement utilisé : CPU.** `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`.
  Sous Windows, le `torch` de PyPI est la version CPU : le SDK se rabat donc sur le CPU en
  `float32`, malgré la RTX 4050. Avantage : les mesures sont représentatives d'un poste sans GPU.
  → À citer dans « Performance analysis » du README.

---

## 2. vocab.json, tokenizer.json et logits

| Question | Réponse |
|---|---|
| Entrées dans `vocab.json` / plus grand id | **151 643** / **151 642** |
| `added_tokens` (dans `tokenizer.json`) | **26**, ids **151 643 → 151 668**, absents de `vocab.json` |
| Total tokens connus | 151 643 + 26 = **151 669** |
| `len(logits)` | **151 936** |

**Écart logits / tokens :** 151 936 − 151 669 = 267 logits sans token associé. La couche de
sortie est dimensionnée en multiple de 128 (151 936 = 1 187 × 128), plus efficace pour le calcul.
Ces logits n'ont pas de sens et doivent être **interdits**.

**Ids utiles :**

| Token | Id |
|---|---|
| `<|im_start|>` | 151644 |
| `<|im_end|>` | 151645 |
| `<think>` | 151667 |
| `</think>` | 151668 |

### Champ `special` des 26 `added_tokens` (`03_vocab.py`)

Aucun des 26 n'est dans `vocab.json` (ni par id, ni par texte).

| Ids | Tokens | `special` |
|---|---|---|
| 151643 – 151645 | `<|endoftext|>` · `<|im_start|>` · `<|im_end|>` | **True** |
| 151646 – 151656 | `<|object_ref_start|>` · `<|object_ref_end|>` · `<|box_start|>` · `<|box_end|>` · `<|quad_start|>` · `<|quad_end|>` · `<|vision_start|>` · `<|vision_end|>` · `<|vision_pad|>` · `<|image_pad|>` · `<|video_pad|>` | **True** |
| 151657 – 151668 | `<tool_call>` · `</tool_call>` · `<|fim_prefix|>` · `<|fim_middle|>` · `<|fim_suffix|>` · `<|fim_pad|>` · `<|repo_name|>` · `<|file_sep|>` · `<tool_response>` · `</tool_response>` · **`<think>`** · **`</think>`** | **False** |

→ **14 True** (151643 – 151656), **12 False** (151657 – 151668).

**Conséquences :**
- `special: True` → `decode` (`skip_special_tokens=True`) les **masque** (`''`). D'où `''` pour
  `<|im_start|>` en section 5.
- `special: False` → `decode` les **affiche** comme du texte. D'où `<think>` visible dans
  l'expérience 2 (section 7).
- **Règle pour la phase 3 :** interdire **tous** les `added_tokens` pendant la génération, quelle
  que soit la valeur de `special`. Sinon une contrainte de string pourrait accepter `<think>`,
  dont chaque caractère est valide dans une string. `special` dit comment `decode` les traite,
  pas s'ils sont autorisés.

---

## 3. Tokens mixtes

- Tokens contenant `"` : **1 344**
- Tokens mélangeant `"` avec `,`, `}` ou `:` : **317 distincts**
  (348 en additionnant les lignes ci-dessous, car certains tokens contiennent deux de ces caractères)

| Caractère | Nb | Exemples (les plus courts) |
|---|---|---|
| `,` | 186 | `",` · `","` · `Ġ",` · `":",` |
| `}` | 88 | `"}` · `"},` · `"}}` · `}\"` |
| `:` | 74 | `":` · `":"` · `Ġ":` · `\":` |

À retenir : `"}}` = exactement la fin d'un appel ; `Ġ"` = espace + guillemet ; `\":` contient
un `\"` qui **ne ferme pas** la string.

---

## 4. Exercice : le JSON reste-t-il valide ?

Situation : le décodeur a écrit `"shrek` dans la valeur de `name`, **dernier** paramètre de
`fn_greet` → la suite attendue est `"}}`.

| Token | Texte obtenu | Valide ? | Pourquoi |
|---|---|---|---|
| `"` | `"shrek"` | ✅ | ferme la string ; début de `"}}` |
| `:"` | `"shrek:"` | ✅ syntaxe, ❌ valeur | `:` est **dans** la string, puis `"` ferme ; valeur `shrek:` |
| `",` | `"shrek",` | ❌ | ferme la string, mais `,` annonce un paramètre qui n'existe pas |
| `"}` | `"shrek"}` | ✅ | ferme la string ; début exact de `"}}` |
| `\"` | `"shrek\"` | ✅ | `"` échappé : la string reste **ouverte** (valeur `shrek"`) |
| `Ġ"` | `"shrek "` | ✅ syntaxe, ❌ valeur | l'espace est **dans** la string, puis `"` ferme |

**Vérification `json.loads` (`04_json_check.py`)** — fins qui rendent le JSON valide :

```
'"'   -> ['}}']
':"'  -> ['}}']
'",'  -> [' "x": 1}}']     ← valide en JSON, mais invente un paramètre : hors schéma
'"}'  -> ['}']
'\\"' -> ['"}}']           ← il faut encore un " : la string était bien ouverte
' "'  -> ['}}']
```

**JSON valide ≠ JSON conforme au schéma.** Le décodeur doit garantir les deux.

**Règle :** un token est accepté si, en lisant ses caractères **un par un** depuis l'état
courant (dans la string, après un `\`, string fermée…), aucun ne viole la grammaire. On n'exige
pas un JSON complet à l'instant t, seulement un **début valide** qui peut être complété aux pas
suivants.

---

## 5. Token spécial `<|im_start|>`

**`encode("<|im_start|>user")`** → 2 ids :

| Id | `decode([id])` |
|---|---|
| 151644 | `''` (token spécial, masqué par `decode`) |
| 872 | `'user'` |

**Contre-épreuve `encode("<|im_start>user")`** (sans le `|` final) → 6 ids :

| Id | `decode([id])` |
|---|---|
| 27 | `'<'` |
| 91 | `'\|'` |
| 318 | `'im'` |
| 4906 | `'_start'` |
| 29 | `'>'` |
| 872 | `'user'` |

151644 a disparu : la chaîne n'est plus reconnue comme token spécial, elle est découpée comme du
texte ordinaire par le BPE.

**Pourquoi c'est important :** le modèle connaît le format de conversation par l'id 151644, pas
par les caractères. Une balise mal écrite ne provoque **aucune erreur** : elle est découpée en
texte ordinaire, la structure n'est plus reconnue et la qualité baisse silencieusement.
**Parade :** définir les balises une seule fois, dans des constantes.

---

## 6. Découpage : `Ġ` et chiffres

Phrase : `"What is the sum of 265 and 345?"`

- **`Ġ`** marque un token **précédé d'un espace** (`Ġis`, `Ġthe`…). `What` n'en a pas (rien
  avant), `?` non plus (collé à `345`).
- **Nombres :** un token **par chiffre** (`2`, `6`, `5`).
- **Espace avant un nombre :** token **à part** (`Ġ`, id 220). Avant le BPE, le pré-découpage de
  Qwen isole chaque chiffre : l'espace ne peut pas s'y coller.

→ Phase 5 : la contrainte de nombre travaille chiffre par chiffre, et doit gérer l'espace (le
refuser, ou forcer `"a":` sans espace).

---

## 7. Génération non contrainte

Question : `Answer only with JSON like {"name": ..., "parameters": {...}}. What is the sum of 2 and 3?`
— greedy, 40 tokens.

### Expérience 1 — prompt brut, sans template

```
The correct answer is 5. But the question is phrased in a way that makes it seem like it's a trick question. So, what is the correct answer?

The correct answer
```

Pas de JSON : le modèle continue un texte au lieu de répondre, et se répète.

### Expérience 2 — chat template

```
<think>
Okay, the user is asking for the sum of 2 and 3. Let me think. The sum of two numbers is just adding them together. So 2 plus 3 equals
```

Il comprend qu'on lui pose une question, mais il « réfléchit » en prose et épuise les 40 tokens.

**Conclusion 1 + 2 :** le chat template améliore la **compréhension** de la tâche, mais le modèle
ne produit toujours pas la **forme** demandée, et le thinking consomme des tokens, donc du temps.

### Expérience 3 — chat template + `<think>\n\n</think>\n\n` vide

````
```json
{"name": "sum", "parameters": {"value": 5}}
```
Human: What is the sum of 2 and 3?
</think>I need to
````

**Avec arrêt sur `<|im_end|>` (151645) :** la sortie s'arrête juste après le bloc JSON. La prose
venait de la boucle, pas du modèle.

| Problème observé | Ce qui le résout |
|---|---|
| Markdown et prose autour du JSON | la structure `{"name": ...}` est **forcée** par le programme, pas générée |
| fonction inventée (`sum`) | le nom est choisi **uniquement** parmi les fonctions définies (`ChoiceConstraint`, phase 4) |
| mauvais paramètres / résultat calculé (`value: 5`) | les clés sont **forcées** depuis la définition, seules les valeurs sont générées (phase 6) |
| pas d'arrêt | le **programme** décide où le JSON se termine (`}}` forcé) ; `<|im_end|>` interdit pendant une valeur |

→ Matière directe pour « Challenges » du README : pourquoi le prompting seul ne suffit pas.

---

## 8. Temps d'un pas

Un appel à `get_logits_from_input_ids`, après warm-up, 5 mesures (CPU) :

| Prompt | Mesures (s) | Moyenne |
|---|---|---|
| 56 tokens | 0.346 · 0.279 · 0.294 · 0.263 · 0.288 | **0,29 s** |
| 441 tokens | 1.179 · 1.070 · 1.101 · 1.094 · 1.090 | **1,11 s** |

- Longueur × ~8 → temps × **3,8** : une partie du coût d'un appel est fixe.
- **Estimation :** 20 tokens générés avec un prompt de ~400 tokens = 20 appels × ~1,1 s
  ≈ **22 s par prompt**. `function_calling_tests.json` contient 11 prompts → **≈ 4 min** + le
  chargement. **Très proche de la limite de 5 min.**
- Leviers (phase 8) : segments forcés ajoutés **sans** appel au modèle ; prompt système court.

---

## 9. Récapitulatif « Terminé quand… »

| À savoir de mémoire | Réponse |
|---|---|
| Taille du vocab | 151 643 (+ 26 added tokens = 151 669 ; 14 `special`, 12 non) |
| Taille des logits | 151 936 |
| Temps de chargement (CPU, cache) | ~2,1 s |
| Durée d'un pas (CPU) | ~0,29 s (56 tokens) · ~1,11 s (441 tokens) |
| Forme de `encode` | tensor `[1, n]` → `[0].tolist()` → `list[int]` |
| Signification de `Ġ` | espace devant le token |
| Exemple de token mixte | `",` · `"}}` · `Ġ":` |
