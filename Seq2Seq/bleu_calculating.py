# -*- coding: utf-8 -*-
from evaluate import get_bleu

preds = [list("It is a guide to action that ensures that the military will forever heed Party commands.".split())]
groud_truths = [list("It is a guide to action which ensures that the military always <unk> the commands of the party.".split())]
print(preds)
print(groud_truths)
print(get_bleu(preds, groud_truths))
"""BUGS
preds:  ['<s>', '<s>', 'I', 'I', 'I', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the', 'of', 'the', 'the', 'the', 'the', 'the']
ground_truths:  ['<s>', '<unk>', 'will', 'they', '<unk>', '<unk>', '<unk>', 'knowledge', 'of', 'both', '<unk>', 'and', '<unk>', '<unk>', 'in', '<unk>', '</s>']
"""