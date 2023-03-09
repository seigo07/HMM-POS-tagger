from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams

class pos_tagger:
    # Define start-of-sentence and end-of-sentence markers in form (word, tag)
    START_OF_SENTENCE_MARKER = "<s>"
    END_OF_SENTENCE_MARKER = "</s>"

    def get_emission_prob(self, sents):

        emission_prob = {}
        emission = [(w.lower(), t) for sent in sents for (w, t) in sent]
        tags = set([t for sent in sents for (_, t) in sent])

        for tag in tags:
            words = [w for (w, t) in emission if t == tag]
            emission_prob[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        return emission_prob

    def get_transition_prob(self, sents):
        transition = []
        for s in sents:
            tags = [t for (w, t) in s]
            transition += ngrams(tags,2)

        tags = set([t for sent in sents for (_, t) in sent])
        transition_prob = {}
        for tag in tags:
            next_tags = [nextTag for (prevTag, nextTag) in transition if prevTag == tag]
            transition_prob[tag] = WittenBellProbDist(FreqDist(next_tags), bins=1e5)

        return transition_prob

    def evaluate(self, sent, pred, comparision):
        # print "word\t\tactual tag\t\tpredited tag"
        for i in range(1, len(sent)-1):
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