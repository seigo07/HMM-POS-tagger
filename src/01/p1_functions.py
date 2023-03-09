from conllu import parse_incr
from nltk import FreqDist

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


# Generate the list of tuples of the word and the part-of-speech.
def get_tagged_sents(sents):
    train_tagged_sents = []
    for sent in sents:
        tagged_sents = []
        for token in sent:
            tagged_sents.append((token['form'], token['upos']))
            # print(token['form'], '->', token['upos'], sep='', end=' ')
        train_tagged_sents.append(tagged_sents)
    return train_tagged_sents


def get_sents(tagged_sents):
    return [[(START_OF_SENTENCE_MARKER, START_OF_SENTENCE_MARKER)]
            + s + [(END_OF_SENTENCE_MARKER, END_OF_SENTENCE_MARKER)]
            for s in tagged_sents]


def change_unknown_words(sents):
    words = []
    for s in sents:
        words += [w for (w, _) in s]
    words_dist = FreqDist(words)
    train_tagged_sents = []
    for sent in sents:
        tagged_sents = []
        for (w, t) in sent:
            word = w
            if words_dist[word] == 1:
                word = "UNK"
            tagged_sents.append((word, t))
        train_tagged_sents.append(tagged_sents)
    return train_tagged_sents


def get_tagged_sents_with_unk(sents, words_sents):
    words = []
    for s in words_sents:
        words += [w for (w, _) in s]
    train_tagged_sents = []
    for sent in sents:
        tagged_sents = []
        for token in sent:
            word = token['form']
            if word not in words:
                word = "UNK"
            tagged_sents.append((word, token['upos']))
        train_tagged_sents.append(tagged_sents)
    return train_tagged_sents

