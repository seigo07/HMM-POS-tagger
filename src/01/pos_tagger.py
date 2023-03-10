from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams
from conllu import parse_incr
import progressbar as progressbar
import time


class posTagger:
    treebank = {'en': '../UD_English-GUM/en_gum', 'fr': '../UD_French-Rhapsodie/fr_rhapsodie',
                'uk': '../UD_Ukrainian-IU/uk_iu'}

    START_OF_SENTENCE_MARKER = "<s>"
    END_OF_SENTENCE_MARKER = "</s>"
    UNKNOWN_WORD_TAG = "UNK"

    train_sents = []
    test_sents = []

    tagset = set()

    emission_prob = {}
    transition_prob = {}

    def __init__(self, lang):

        treebank = self.train_corpus(lang)
        initial_train_sents = self.conllu_corpus(treebank)
        initial_test_sents = self.conllu_corpus(treebank)

        train_tagged_sents = self.get_tagged_sents(initial_train_sents)
        self.train_sents = self.get_sents_with_markers(train_tagged_sents)

        # Using UNK tag
        # self.set_train_tagged_sents_with_unk()

        test_tagged_sents = self.get_tagged_sents(initial_test_sents)
        # Using UNK tag
        # test_tagged_sents = self.get_tagged_sents_with_unk(initial_test_sents)
        self.test_sents = self.get_sents_with_markers(test_tagged_sents)

        self.set_tagset(self.test_sents)

        # Estimate the emission and transition probabilities
        print("Step 1: Estimating probabilities")
        self.set_emission_prob()
        self.set_transition_prob()

    def train_corpus(self, lang):
        return self.treebank[lang] + '-ud-train.conllu'

    def test_corpus(self, lang):
        return self.treebank[lang] + '-ud-test.conllu'

    def prune_sentence(self, sent):
        return [token for token in sent if type(token['id']) is int]

    def conllu_corpus(self, path):
        data_file = open(path, 'r', encoding='utf-8')
        sents = list(parse_incr(data_file))
        return [self.prune_sentence(sent) for sent in sents]

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

    def set_train_tagged_sents_with_unk(self):
        words = [w for s in self.train_sents for (w, _) in s]
        words_dist = FreqDist(words)
        tagged_sents = []
        for sent in self.train_sents:
            tagged_sent = []
            for (w, t) in sent:
                word = w
                # if word.endswith('ing'):
                if words_dist[word] == 1:
                    word = self.UNKNOWN_WORD_TAG
                tagged_sent.append((word, t))
            tagged_sents.append(tagged_sent)
        self.train_sents = tagged_sents

    def get_tagged_sents_with_unk(self, sents):
        words = [w for s in self.train_sents for (w, _) in s]
        tagged_sents = []
        for sent in sents:
            tagged_sent = []
            for token in sent:
                word = token['form']
                # if word.endswith('ing'):
                if word not in words:
                    word = self.UNKNOWN_WORD_TAG
                tagged_sent.append((word, token['upos']))
            tagged_sents.append(tagged_sent)
        return tagged_sents

    def set_emission_prob(self):

        # P(word|tag) = emissionProb[tag].prob(word)
        emission = []
        for s in self.train_sents:
            emission += [(w.lower(), t) for (w, t) in s] # treat for both lowercase and uppercase in the same way

        emission_prob = {}
        tags = set([t for sent in self.train_sents for (_, t) in sent])
        for tag in tags:
            words = [w for (w, t) in emission if t == tag]
            emission_prob[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        self.emission_prob = emission_prob
        # tags = set([t for sent in self.train_sents for (_, t) in sent])
        #
        # for tag in tags:
        #     words = [w.lower() for sent in self.train_sents for (w, t) in sent if t == tag]
        #     self.emission_prob[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

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
        # print "word\t\tactual tag\t\tpredited tag"
        for i in range(1, len(sent) - 1):
            # print sent[i][0]+"\t\t"+sent[i][1]+"\t\t"+pred[i][1]
            assert (sent[i][0].lower() == pred[i][0])

            if sent[i][1] == self.START_OF_SENTENCE_MARKER or sent[i][1] == self.END_OF_SENTENCE_MARKER:
                continue

            # if pred[i][1] == self.startWord or pred[i][1] == self.endWord:
            #     comparision["extra"] += 1

            if sent[i][1] == pred[i][1]:
                comparision["total"]["correct"] += 1
                comparision[sent[i][1]]["correct"] += 1
                # if i==1:
                #     comparision["first"]["correct"] += 1
                # if i==(len(sent)-2):
                #     comparision["last"]["correct"] += 1
            else:
                comparision["total"]["incorrect"] += 1
                comparision[sent[i][1]]["incorrect"] += 1
                # if i==1:
                #     comparision["first"]["incorrect"] += 1
                # if i==(len(sent)-2):
                #     comparision["last"]["incorrect"] += 1

        return comparision

    def set_tagset(self, sents):
        for s in sents:
            t = [t for (w, t) in s]
            self.tagset.update(t)

    def initialse(self, sentence):
        vtb = {}
        for tag in self.tagset:
            vtb[tag] = [None for w in sentence] + [None]

        # initialise
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

    def finalise(self, vtb, sentence):
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
        vtb = self.initialse(sentence)
        vtb = self.run(vtb, sentence)
        maxProb = self.finalise(vtb, sentence)

        if maxProb[0] is None:
            return [(w, None) for w in sentence]

        return self.backtrack(vtb, sentence, maxProb)

    def start(self):

        start_time = time.time()
        leaning_time = time.time()
        print("Training time", (leaning_time - start_time))
        print("Step 2: Applying HMM")
        # Applying a trained HMM on sentences from the testing data
        bar = progressbar.ProgressBar(maxval=len(self.test_sents),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        comparision = {
            "total": {"correct": 0, "incorrect": 0},
            "first": {"correct": 1, "incorrect": 0},
            "last": {"correct": 1, "incorrect": 0},
        }
        for tag in self.tagset:
            comparision[tag] = {"correct": 0, "incorrect": 0}

        index = 0

        cc = 0
        for sent in self.test_sents:
            bar.update(index)
            if len(sent) > 100:
                continue
            words = [w.lower() for (w, t) in sent]
            Xs = [w for (w, t) in sent if t == "X"]
            if len(Xs) > 0:
                cc += 1
                continue

            predicted = self.apply(words)
            comparision = self.evaluate(sent, predicted, comparision)
            index += 1
        bar.finish()
        print(cc)
        predicting_time = time.time()
        print("Predicting time", (predicting_time - leaning_time))
        print("Step 3: Evaluation")
        # Evaluation: comparing them with the gold-standard sequence of tags for that sentence
        for tag in self.tagset:
            if tag == self.START_OF_SENTENCE_MARKER or tag == self.END_OF_SENTENCE_MARKER:
                continue

            if (comparision[tag]["correct"] + comparision[tag]["incorrect"]) == 0:
                print("Tag {0} {1}".format(tag, "Inf"))
                continue

            accuracy = comparision[tag]["correct"] * 100.0 / (
                    comparision[tag]["correct"] + comparision[tag]["incorrect"])
            print("Tag {0} {1:.2f}".format(tag, accuracy))

        accuracy = comparision["total"]["correct"] * 100.0 / (
                comparision["total"]["correct"] + comparision["total"]["incorrect"])
        print("Total {0:.2f}".format(accuracy))
