from io import open

from conllu import parse_incr

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

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
print(len(train_sents), 'training sentences')
print(len(test_sents), 'test sentences')

# Illustration how to access the word and the part-of-speech of tokens.
for sent in train_sents:
	for token in sent:
		print(token['form'], '->', token['upos'], sep='', end=' ')
	print()
