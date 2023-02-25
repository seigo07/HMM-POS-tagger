from io import open

import inline as inline
import matplotlib as matplotlib
import nltk
from conllu import parse_incr
from collections import Counter

treebank = {}
treebank['en'] = 'UD_English-GUM/en_gum'
treebank['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
treebank['uk'] = 'UD_Ukrainian-IU/uk_iu'


def train_corpus(lang):
    return treebank[lang] + '-ud-train.conllu'


def test_corpus(lang):
    return treebank[lang] + '-ud-test.conllu'


# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


# Choose language.
lang = 'en'
# lang = 'fr'
# lang = 'uk'

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
# print(len(train_sents), 'training sentences')
# print(len(test_sents), 'test sentences')

# Illustration how to access the word and the part-of-speech of tokens.
# for sent in train_sents:
# 	for token in sent:
# 		print(token['form'], '->', token['upos'], sep='', end=' ')
# 	print()

# The number of each tag
n_tags = Counter(token['upos'] for sent in train_sents for token in sent)
n_words = Counter(token['form'] for sent in train_sents for token in sent)


# print(n_words)

# tag_fd = nltk.FreqDist(token['upos'] for sent in train_sents for token in sent)
# print("tag_fd = ", tag_fd.most_common(), end = '')
# tag_fd.plot(cumulative = True);

# word_tag_pairs = nltk.bigrams(train_sents)
# noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
# fdist = nltk.FreqDist(noun_preceders)
# print([tag for (tag, _) in fdist.most_common()], end = '')

