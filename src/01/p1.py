import time
import sys
import progressbar as progressbar

from p1_functions import *
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
start_time = time.time()

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))

train_tagged_sents = get_tagged_sents(train_sents)
train_sents = get_sents(train_tagged_sents)
# train_sents = change_unknown_words(train_sents)

test_tagged_sents = get_tagged_sents(test_sents)
# test_tagged_sents = get_tagged_sents_with_unk(test_sents, train_sents)
test_sents = get_sents(test_tagged_sents)

tagger = viterbi_pos_tagger()
tagger.set_tagset(train_sents)

# Estimate the emission and transition probabilities
print("Step 1: Estimating probabilities")
emission_prob = tagger.get_emission_prob(train_sents, tagger.tagset)
transition_prob = tagger.get_transition_prob(train_sents, tagger.tagset)

leaning_time = time.time()
print("Training time", (leaning_time - start_time))
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
for tag in tagger.tagset:
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

    predicted = tagger.apply(words, emission_prob, transition_prob)
    comparision = tagger.evaluate(sent, predicted, comparision)
    index += 1
bar.finish()
print(cc)
predicting_time = time.time()
print("Predicting time", (predicting_time - leaning_time))
print("Step 3: Evaluation")
# Evaluation: comparing them with the gold-standard sequence of tags for that sentence
# tagger.set_tag_set(test_sents)
for tag in tagger.tagset:
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
