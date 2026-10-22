Explorer vocab.json et tokenizer.json

[ ] Combien d'entrées dans vocab.json ? Quel est le plus grand id ?
    - Len de vocab = 151643 / Plus grand id = 151642

[ ] Combien de added_tokens ? Quels sont leurs ids, et sont-ils dans vocab.json ?
    - 26 added tokens / Added tokens : de 151643 a 151668 / Non

[ ] Additionner les deux et comparer avec len(logits). Que représentent les ids restants ?
    - 26 + 151643 = 151669 / len(logits) = 151936. Il y a des logits sans token associes. Cela vient du fait que le calcule est en multiple de 128 pour generer les logits, plus efficace pour le GPU, ce qui ne correspond pas forcement aux nombres de token. Les logits en surplus n'ont pas de sens et doivent etre interdit.

[ ] Retrouver dans added_tokens les ids de <|im_start|>, <|im_end|>, <think> et </think> : ils serviront pour le format de conversation (étape suivante).
    - im_start -> 151644
    - im_end -> 151645
    - <think> -> 151667
    - </think> -> 151668

[ ] Combien de tokens contiennent " ?


[ ] Lesquels mélangent " avec ,, } ou : ? Relève 2 ou 3 exemples.
