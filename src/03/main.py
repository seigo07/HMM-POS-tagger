'''
Created on 28 Feb 2018
'''
from hmm import HMM
from tagger import pos_tagger
from nltk.corpus import brown, conll2000, treebank
import sys
import time
from nltk.corpus import brown

from corpus import *

CONLL2000 = 1
CONLL2000_UNIVERSAL = 2
TREEBANK = 3
TREEBANK_UNIVERSAL = 4
BROWN_UNIVERSAL = 5

INVALID_ARGS_NUMBER_ERROR = "Usage: python p1 <lang>"
INVALID_ARGS_LANG_ERROR = "Usage: en, fr, or uk for <lang>"

VALID_LANG_LIST = ["en", "fr", "uk"]
ARGV_NUMBER = 2


def get_corpus(selected_corpus):
    # get the right corpus and tagset
    tagset = ""
    corpus_name = ""
    if selected_corpus == CONLL2000_UNIVERSAL:
        corpus = conll2000
        tagset = "universal"
        corpus_name = "Conll2000"
    elif selected_corpus == CONLL2000:
        corpus = conll2000
        corpus_name = "Conll2000"
    elif selected_corpus == TREEBANK:
        corpus = treebank
        corpus_name = "Treebank"
    elif selected_corpus == BROWN_UNIVERSAL:
        corpus = brown
        tagset = "universal"
        corpus_name = "Brown"
    elif selected_corpus == TREEBANK_UNIVERSAL:
        corpus = treebank
        tagset = "universal"
        corpus_name = "Treebank"
    else:
        print("corpus unavailable")
        quit() 
    return corpus, tagset, corpus_name


# Check <lang> is valid
def is_lang_valid():
    return sys.argv[1] in VALID_LANG_LIST

if __name__ == '__main__':

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
    train_tagged_sents = get_tagged_sents(train_sents)
    test_tagged_sents = get_tagged_sents(test_sents)

    # train HMM
    hmm = HMM(train_tagged_sents, test_tagged_sents)
    
    # pos tagging with viterbi algorithm 
    # tagger = pos_tagger(hmm)
    # tagger.viterbi()
    #
    # # print confusion matrix
    # print("")
    # # if tagset == "universal":
    # #     tagger.print_confusion_matrix()
    # # else:
    # tagger.print_summary()
    # print("")
    #
    # # print overall accuracy
    # print("overall accuracy, correct: %d/%d percentage: %0.2f \n" % \
    #   (tagger.overall_accuracy["correct"], tagger.overall_accuracy["words"], tagger.overall_accuracy["percentage"]))