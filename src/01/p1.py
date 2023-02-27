import time
import sys
import progressbar as progressbar
from nltk.corpus import brown

from corpus import *
from viterbi_pos_tagger import viterbi_pos_tagger

INVALID_ARGS_NUMBER_ERROR = "Usage: python p1 <lang>"
INVALID_ARGS_LANG_ERROR = "Usage: en, fr, or uk for <lang>"

VALID_LANG_LIST = ["en", "fr", "uk"]
ARGV_NUMBER = 2

# Check <lang> is valid
def is_lang_valid():
    return sys.argv[1] in VALID_LANG_LIST


# Validate the number of args in arg.
if len(sys.argv) != ARGV_NUMBER:
    exit(INVALID_ARGS_NUMBER_ERROR)

# Validate language in arg.
if not is_lang_valid():
    exit(INVALID_ARGS_LANG_ERROR)

lang = sys.argv[1]
print("Load...")
startTime = time.time()

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
# print(len(train_sents), 'training sentences')
# print(len(test_sents), 'test sentences')

train_tagged_sents = get_tagged_sents(train_sents)
train_sents = get_sents(train_tagged_sents)
# print(train_tagged_sents)
# print(train_sents)

test_tagged_sents = get_tagged_sents(test_sents)
test_sents = get_sents(test_tagged_sents)
# print(test_tagged_sents)
# print(test_sents)

# brown test
# data = brown.tagged_sents(tagset='universal')[0:100]
# data = brown.tagged_sents(tagset='universal')
# sents = [[(START_OF_SENTENCE_MARKER, START_OF_SENTENCE_MARKER)] + s + [(END_OF_SENTENCE_MARKER, END_OF_SENTENCE_MARKER)] for s in data]
# traingRation = 0.95
# train_sents = test_sents[0: int(traingRation * len(test_sents))]
# test_sents = test_sents[int(traingRation * len(test_sents)):]

tagger = viterbi_pos_tagger()
tagger.set_tag_set(train_sents)

# Estimate the emission and transition probabilities
# Estimate the emission probabilities
print("Step 1: Estimating probabilities")
emissionProb = tagger.getEmissionProb(train_sents, tagger.tag_set)
transitionProb = tagger.getTransitionProb(train_sents, tagger.tag_set)
# print("emissionProb:", emissionProb)
# print("transitionProb:", transitionProb)
leaningTime = time.time()
print("Training time", (leaningTime - startTime))
print("Step 2: Applying HMM")
# Applying a trained HMM on sentences from the testing data
bar = progressbar.ProgressBar(maxval=len(test_sents),
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

comparision = {
    "total": {"correct": 0, "incorrect": 0},
    "first": {"correct": 1, "incorrect": 0},
    "last": {"correct": 1, "incorrect": 0},
}
for tag in tagger.tag_set:
    comparision[tag] = {"correct": 0, "incorrect": 0}

index = 0

cc = 0
for sent in test_sents:
    bar.update(index)
    if len(sent) > 100:
        continue
    words = [w.lower() for (w, t) in sent]
    Xs = [w for (w, t) in sent if t == "X"]
    if len(Xs) > 0:
        cc += 1
        continue

    predicted = tagger.apply(words, emissionProb, transitionProb)
    comparision = tagger.evaluate(sent, predicted, comparision)
    index += 1
bar.finish()
print(cc)
predictingTime = time.time()
print("Predicting time", (predictingTime - leaningTime))
print("Step 3: Evaluation")
# Evaluation: comparing them with the gold-standard sequence of tags for that sentence
# tagger.set_tag_set(test_sents)
for tag in tagger.tag_set:
    if tag == START_OF_SENTENCE_MARKER or tag == END_OF_SENTENCE_MARKER:
        continue

    if (comparision[tag]["correct"] + comparision[tag]["incorrect"]) == 0:
        print("Tag {0} {1}".format(tag, "Inf"))
        continue

    accuracy = comparision[tag]["correct"] * 100.0 / (comparision[tag]["correct"] + comparision[tag]["incorrect"])
    print("Tag {0} {1:.2f}".format(tag, accuracy))

accuracy = comparision["total"]["correct"] * 100.0 / (
        comparision["total"]["correct"] + comparision["total"]["incorrect"])
print("Total {0:.2f}".format(accuracy))
