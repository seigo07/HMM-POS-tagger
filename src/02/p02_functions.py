from conllu import parse_incr
from nltk import WittenBellProbDist, ngrams, FreqDist

treebank = {}
treebank['en'] = '../UD_English-GUM/en_gum'
treebank['fr'] = '../UD_French-Rhapsodie/fr_rhapsodie'
treebank['uk'] = '../UD_Ukrainian-IU/uk_iu'

START_OF_SENTENCE_MARKER = "<s>"
END_OF_SENTENCE_MARKER = "</s>"


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


def get_tagged_words(sents):
    tagged_words = []
    for sent in sents:
        for token in sent:
            tagged_words.append(token['upos'])

    return tagged_words


# Generate the list of tuples of the word and the part-of-speech.
def get_tagged_sents(sents):
    train_tagged_sents = []
    for sent in sents:
        for token in sent:
            train_tagged_sents.append([(token['form'], token['upos'])])
            # print(token['form'], '->', token['upos'], sep='', end=' ')

    return train_tagged_sents


def get_tagged_sents2(sents):
    tagged_sents = []
    for sent in sents:
        sents = []
        for token in sent:
            sents.append((token['form'], token['upos']))
            # print(token['form'], '->', token['upos'], sep='', end=' ')
        tagged_sents.append(sents)
    return tagged_sents


# Generate the list of tuples of the word and the part-of-speech.
def get_untagged_sents(sents):
    untagged_sents = []
    for sent in sents:
        sents = []
        for token in sent:
            sents.append(token['form'])
            # print(token['form'], '->', token['upos'], sep='', end=' ')
        untagged_sents.append(sents)
    return untagged_sents


def get_sents(tagged_sents):
    return [[(START_OF_SENTENCE_MARKER, START_OF_SENTENCE_MARKER)]
            + s + [(END_OF_SENTENCE_MARKER, END_OF_SENTENCE_MARKER)]
            for s in tagged_sents]

