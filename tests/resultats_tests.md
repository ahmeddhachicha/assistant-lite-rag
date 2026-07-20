# Résultats des tests

Modèle : `qwen2.5:1.5b` — corpus : `corpus_de_travail.txt`

## Comparaison des configurations

Paramètre modifié entre les deux configurations : **nombre de segments récupérés** (3 vs 6), taille des segments fixe (800).

| Config | Question | Réponse attendue | Statut | Segments | Temps (s) | Réponse (extrait) |
|---|---|---|---|---|---|---|
| Config A (k=3) | Que dit le discours sur le changement climatique ? | présente | ok | 3 | 1.83 | Le discours met l'accent sur la nécessité pour les Américains d'agir face au changement climatique. Il souligne que non … |
| Config A (k=3) | Que dit le discours sur Medicare et la Sécurité sociale ? | présente | ok | 3 | 1.9 | Les discours mentionnent que Medicare et la Sécurité sociale sont des systèmes qui assurent un minimum de sécurité et de… |
| Config A (k=3) | Quel est le prix du bitcoin mentionné dans le discours ? | absente | hors_corpus | 3 | 0.47 | Je ne trouve cette information dans le corpus.… |
| Config B (k=6) | Que dit le discours sur le changement climatique ? | présente | ok | 6 | 1.69 | Nous allons répondre à la menace du changement climatique, sachant que le fait de ne pas le faire serait trahir nos enfa… |
| Config B (k=6) | Que dit le discours sur Medicare et la Sécurité sociale ? | présente | ok | 6 | 2.88 | Nous reconnaissons que, quel que soit le degré de responsabilité avec lequel nous vivons nos vies, n'importe lequel d'en… |
| Config B (k=6) | Quel est le prix du bitcoin mentionné dans le discours ? | absente | hors_corpus | 6 | 1.35 | Je ne trouve cette information dans le corpus.… |
