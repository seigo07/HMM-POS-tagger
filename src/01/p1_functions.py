from conllu import parse_incr
from nltk import FreqDist

treebank = {}
treebank['en'] = '../UD_English-GUM/en_gum'
treebank['fr'] = '../UD_French-Rhapsodie/fr_rhapsodie'
treebank['uk'] = '../UD_Ukrainian-IU/uk_iu'

START_OF_SENTENCE_MARKER = "<s>"
END_OF_SENTENCE_MARKER = "</s>"
UNKNOWN_WORD_TAG = "UNK"

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
    tagged_sents = []
    for sent in sents:
        tagged_sent = []
        for token in sent:
            tagged_sent.append((token['form'], token['upos']))
        tagged_sents.append(tagged_sent)
    return tagged_sents


def get_sents_with_markers(tagged_sents):
    return [[(START_OF_SENTENCE_MARKER, START_OF_SENTENCE_MARKER)]
            + s + [(END_OF_SENTENCE_MARKER, END_OF_SENTENCE_MARKER)]
            for s in tagged_sents]


def get_train_tagged_sents_with_unk(sents):
    words = [w for s in sents for (w, _) in s]
    words_dist = FreqDist(words)
    tagged_sents = []
    for sent in sents:
        tagged_sent = []
        for (w, t) in sent:
            word = w
            # if word.endswith('ing'):
            if words_dist[word] == 1:
                word = UNKNOWN_WORD_TAG
            tagged_sent.append((word, t))
        tagged_sents.append(tagged_sent)
    return tagged_sents


def get_test_tagged_sents_with_unk(sents, sents_with_unk):
    words = [w for s in sents_with_unk for (w, _) in s]
    tagged_sents = []
    for sent in sents:
        tagged_sent = []
        for token in sent:
            word = token['form']
            # if word.endswith('ing'):
            if word not in words:
                word = UNKNOWN_WORD_TAG
            tagged_sent.append((word, token['upos']))
        tagged_sents.append(tagged_sent)
    return tagged_sents

