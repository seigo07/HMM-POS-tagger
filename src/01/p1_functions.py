from conllu import parse_incr

treebank = {}
treebank['en'] = 'UD_English-GUM/en_gum'
treebank['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
treebank['uk'] = 'UD_Ukrainian-IU/uk_iu'

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


def get_emission_sents(sents):
    emission_sents = []
    for sent in sents:
        for index in range(len(sent)):
            emission_sents.append(sent[index]['upos']+', '+sent[index]['form'])
    return emission_sents


def get_transition_sents(sents):
    transition_sents = []
    for sent in sents:
        if len(sent) != 1:
            for index in range(len(sent)):
                if sent[index]['id'] == 1:
                    transition_sents.append('<s>' + ', ' + sent[index]['upos'])
                elif sent[index]['id'] == len(sent):
                    transition_sents.append(sent[index - 1]['upos'] + ', ' + sent[index]['upos'])
                    transition_sents.append(sent[index]['upos'] + ', ' + '</s>')
                else:
                    transition_sents.append(sent[index - 1]['upos'] + ', ' + sent[index]['upos'])
        else:
            transition_sents.append('<s>' + ', ' + sent[0]['upos'])
            transition_sents.append(sent[0]['upos'] + ', ' + '</s>')
    return transition_sents


def get_tagset(sents):
    tagset = set()
    for tags in sents:
        t = tags[:tags.find(", ")]
        tagset.add(t)
    return tagset
