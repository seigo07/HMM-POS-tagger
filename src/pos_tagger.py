from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams
from conllu import parse_incr


class posTagger:
    treebank = {'en': '../UD_English-GUM/en_gum', 'fr': '../UD_French-Rhapsodie/fr_rhapsodie',
                'uk': '../UD_Ukrainian-IU/uk_iu'}

    START_OF_SENTENCE_MARKER = "<s>"
    END_OF_SENTENCE_MARKER = "</s>"
    UNKNOWN_WORD_TAG = "UNK"

    lang = 'en'
    train_sents = []
    test_sents = []
    tagset = set()
    emission_prob = {}
    transition_prob = {}

    def __init__(self, lang):

        self.lang = lang

        train_treebank = self.train_corpus()
        test_treebank = self.test_corpus()
        initial_train_sents = self.conllu_corpus(train_treebank)
        initial_test_sents = self.conllu_corpus(test_treebank)

        train_tagged_sents = self.get_tagged_sents(initial_train_sents)
        self.train_sents = self.get_sents_with_markers(train_tagged_sents)

        # Using UNK tag for training
        self.set_train_tagged_sents_with_unk_train()

        test_tagged_sents = self.get_tagged_sents(initial_test_sents)
        self.test_sents = self.get_sents_with_markers(test_tagged_sents)

        # Using UNK tag for testing
        self.set_train_tagged_sents_with_unk_test()

        self.set_tagset(self.test_sents)

        # Estimate the emission and transition probabilities
        print(":::::::::::::::::::::::::::Step 1: Estimating probabilities:::::::::::::::::::::::::::")
        self.set_emission_prob()
        self.set_transition_prob()

    def train_corpus(self):
        return self.treebank[self.lang] + '-ud-train.conllu'

    def test_corpus(self):
        return self.treebank[self.lang] + '-ud-test.conllu'

    def prune_sentence(self, sent):
        return [token for token in sent if type(token['id']) is int]

    def conllu_corpus(self, path):
        data_file = open(path, 'r', encoding='utf-8')
        sents = list(parse_incr(data_file))
        return [self.prune_sentence(sent) for sent in sents]

    def set_tagset(self, sents):
        for s in sents:
            t = [t for (w, t) in s]
            self.tagset.update(t)

    def get_tagged_sents(self, sents):
        tagged_sents = []
        for sent in sents:
            tagged_sent = []
            for token in sent:
                tagged_sent.append((token['form'], token['upos']))
            tagged_sents.append(tagged_sent)
        return tagged_sents

    def get_sents_with_markers(self, tagged_sents):
        return [[(self.START_OF_SENTENCE_MARKER, self.START_OF_SENTENCE_MARKER)]
                + s + [(self.END_OF_SENTENCE_MARKER, self.END_OF_SENTENCE_MARKER)]
                for s in tagged_sents]

    def set_train_tagged_sents_with_unk_train(self):
        words = [w for s in self.train_sents for (w, _) in s]
        words_dist = FreqDist(words)
        tagged_sents = []
        for sent in self.train_sents:
            tagged_sent = []
            for index, (w, t) in enumerate(sent):
                word = self.check_unk_word_train(sent, index, words_dist)
                tagged_sent.append((word, t))
            tagged_sents.append(tagged_sent)
        self.train_sents = tagged_sents

    def set_train_tagged_sents_with_unk_test(self):
        train_words = [w for s in self.train_sents for (w, _) in s]
        words_dist = FreqDist(train_words)
        tagged_sents = []
        for sent in self.test_sents:
            tagged_sent = []
            for index, (w, t) in enumerate(sent):
                word = self.check_unk_word_test(sent, index, train_words, words_dist)
                tagged_sent.append((word, t))
            tagged_sents.append(tagged_sent)
        self.test_sents = tagged_sents

    def check_unk_word_train(self, sent, index, words_dist):
        word = sent[index][0]
        if words_dist[word] == 1:
            return self.check_unk_pattern(sent, index)
        return word

    def check_unk_word_test(self, sent, index, train_words, words_dist):
        word = sent[index][0]
        if word not in train_words:
            return self.check_unk_pattern(sent, index)
        return self.check_unk_word_train(sent, index, words_dist)

    def check_unk_pattern(self, sent, index):
        word = sent[index][0]
        is_first = index == 1
        if self.lang == 'en':
            # proper noun
            if not is_first and word.istitle():
                return self.UNKNOWN_WORD_TAG + "-propn"
            # verb or noun (gerund)
            elif word.endswith('ing'):
                return self.UNKNOWN_WORD_TAG + "-ing"
            # verb or adjective
            elif word.endswith('ed'):
                return self.UNKNOWN_WORD_TAG + "-ed"
            # noun
            elif word.endswith('ment'):
                return self.UNKNOWN_WORD_TAG + "-ment"
            # noun
            elif word.endswith('ness'):
                return self.UNKNOWN_WORD_TAG + "-ness"
            # verb
            elif word.endswith('ify'):
                return self.UNKNOWN_WORD_TAG + "-ify"
            # verb
            elif word.endswith('ize'):
                return self.UNKNOWN_WORD_TAG + "-ize"
            # adjective
            elif word.endswith('able'):
                return self.UNKNOWN_WORD_TAG + "-able"
            # adjective
            elif word.endswith('ful'):
                return self.UNKNOWN_WORD_TAG + "-ful"
            # adverb
            elif word.endswith('ly'):
                return self.UNKNOWN_WORD_TAG + "-ly"
            else:
                return word
                # catch-all
                # return self.UNKNOWN_WORD_TAG
        elif self.lang == 'fr':
            # proper noun
            if not is_first and word.istitle():
                return self.UNKNOWN_WORD_TAG + "-propn"
            # noun
            elif word.endswith('ion'):
                return self.UNKNOWN_WORD_TAG + "-ion"
            # noun
            elif word.endswith('ison'):
                return self.UNKNOWN_WORD_TAG + "-ison"
            # noun
            elif word.endswith('eur'):
                return self.UNKNOWN_WORD_TAG + "-eur"
            # noun
            elif word.endswith('age'):
                return self.UNKNOWN_WORD_TAG + "-age"
            # noun
            elif word.endswith('ine'):
                return self.UNKNOWN_WORD_TAG + "-ine"
            # verb
            elif word.endswith('er'):
                return self.UNKNOWN_WORD_TAG + "-er"
            # verb
            elif word.endswith('ir'):
                return self.UNKNOWN_WORD_TAG + "-ir"
            # adjective
            elif word.endswith('ble'):
                return self.UNKNOWN_WORD_TAG + "-ble"
            # adverb
            elif word.endswith('ment'):
                return self.UNKNOWN_WORD_TAG + "-ment"
            else:
                return word
                # catch-all
                # return self.UNKNOWN_WORD_TAG
        elif self.lang == 'uk':
            # proper noun
            if not is_first and word.istitle():
                return self.UNKNOWN_WORD_TAG + "-propn"
            # adjective
            elif word.endswith('ий'):
                return self.UNKNOWN_WORD_TAG + "-ий"
            # adjective
            elif word.endswith('им'):
                return self.UNKNOWN_WORD_TAG + "-им"
            # adjective
            elif word.endswith('их'):
                return self.UNKNOWN_WORD_TAG + "-их"
            # adjective
            elif word.endswith('ому'):
                return self.UNKNOWN_WORD_TAG + "-ому"
            # adjective
            elif word.endswith('ого'):
                return self.UNKNOWN_WORD_TAG + "-ого"
            # adjective
            elif word.endswith('ої'):
                return self.UNKNOWN_WORD_TAG + "-ої"
            # noun
            elif word.endswith('ик'):
                return self.UNKNOWN_WORD_TAG + "-ик"
            # noun
            elif word.endswith('ець'):
                return self.UNKNOWN_WORD_TAG + "-ець"
            # noun
            elif word.endswith('ість'):
                return self.UNKNOWN_WORD_TAG + "-ість"
            else:
                return word
                # catch-all
                # return self.UNKNOWN_WORD_TAG

    def set_emission_prob(self):

        tags = set([t for sent in self.train_sents for (_, t) in sent])

        for tag in tags:
            words = [w.lower() for sent in self.train_sents for (w, t) in sent if t == tag]
            self.emission_prob[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    def set_transition_prob(self):
        transition = []
        for s in self.train_sents:
            tags = [t for (w, t) in s]
            transition += ngrams(tags, 2)

        tags = set([t for sent in self.train_sents for (_, t) in sent])
        for tag in tags:
            next_tags = [nextTag for (prevTag, nextTag) in transition if prevTag == tag]
            self.transition_prob[tag] = WittenBellProbDist(FreqDist(next_tags), bins=1e5)

    def evaluate(self, sent, pred, comparision):
        for i in range(1, len(sent) - 1):
            assert (sent[i][0].lower() == pred[i][0])

            if sent[i][1] == self.START_OF_SENTENCE_MARKER or sent[i][1] == self.END_OF_SENTENCE_MARKER:
                continue

            if sent[i][1] == pred[i][1]:
                comparision["total"]["correct"] += 1
                comparision[sent[i][1]]["correct"] += 1
            else:
                comparision["total"]["incorrect"] += 1
                comparision[sent[i][1]]["incorrect"] += 1

        return comparision

    def initialization(self, sentence):
        vtb = {}
        for tag in self.tagset:
            vtb[tag] = [None for w in sentence] + [None]

        for tag in self.tagset:
            word = sentence[1]
            p = self.transition_prob[self.START_OF_SENTENCE_MARKER].prob(tag) * self.emission_prob[tag].prob(word)
            vtb[tag][1] = (self.START_OF_SENTENCE_MARKER, p)

        return vtb

    def run(self, vtb, sentence):
        for i in range(2, len(sentence) - 1):
            word = sentence[i]
            for tag in self.tagset:
                max_prob = (None, 0)
                for prevTag in self.tagset:
                    p = vtb[prevTag][i - 1][1] * self.transition_prob[prevTag].prob(tag) * self.emission_prob[tag].prob(
                        word)
                    if p > max_prob[1]:
                        max_prob = (prevTag, p)

                vtb[tag][i] = max_prob

        return vtb

    def determine(self, vtb, sentence):
        max_prob = (None, 0)
        n = len(sentence) - 1
        for tag in self.tagset:
            p = vtb[tag][n - 1][1] * self.transition_prob[tag].prob(self.END_OF_SENTENCE_MARKER)
            if p > max_prob[1]:
                max_prob = (tag, p)

        return max_prob

    def backtrack(self, vtb, sentence, max_prob):
        predicted_pod = [(self.END_OF_SENTENCE_MARKER, self.END_OF_SENTENCE_MARKER)]
        for i in range(len(sentence) - 2, 0, -1):
            word = sentence[i]
            tag = max_prob[0]

            predicted_pod.insert(0, (word, tag))
            max_prob = vtb[tag][i]

        predicted_pod.insert(0, (self.START_OF_SENTENCE_MARKER, self.START_OF_SENTENCE_MARKER))
        return predicted_pod

    def apply(self, sentence):
        vtb = self.initialization(sentence)
        vtb = self.run(vtb, sentence)
        maxProb = self.determine(vtb, sentence)

        if maxProb[0] is None:
            return [(w, None) for w in sentence]

        return self.backtrack(vtb, sentence, maxProb)

    def start(self):

        print(":::::::::::::::::::::::::::Step 2: Applying HMM:::::::::::::::::::::::::::")
        comparision = {
            "total": {"correct": 0, "incorrect": 0},
            "first": {"correct": 1, "incorrect": 0},
            "last": {"correct": 1, "incorrect": 0},
        }
        for tag in self.tagset:
            comparision[tag] = {"correct": 0, "incorrect": 0}

        for sent in self.test_sents:
            if len(sent) > 100:
                continue
            words = [w.lower() for (w, t) in sent]
            Xs = [w for (w, t) in sent if t == "X"]
            if len(Xs) > 0:
                continue

            predicted = self.apply(words)
            comparision = self.evaluate(sent, predicted, comparision)

        print(":::::::::::::::::::::::::::Step 3: Evaluation:::::::::::::::::::::::::::")
        for tag in self.tagset:
            if tag == self.START_OF_SENTENCE_MARKER or tag == self.END_OF_SENTENCE_MARKER:
                continue

            if (comparision[tag]["correct"] + comparision[tag]["incorrect"]) == 0:
                print("{0} {1}".format(tag, "Inf"))
                continue

            accuracy = comparision[tag]["correct"] * 100.0 / (
                    comparision[tag]["correct"] + comparision[tag]["incorrect"])
            print("{0} {1:.2f}".format(tag, accuracy))

        accuracy = comparision["total"]["correct"] * 100.0 / (
                comparision["total"]["correct"] + comparision["total"]["incorrect"])
        print("Overall {0:.2f}".format(accuracy))
