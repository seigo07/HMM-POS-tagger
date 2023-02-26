from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams

class pos_tagger:
    # Define start-of-sentence and end-of-sentence markers in form (word, tag)
    startWord = "<s>"
    endWord = "</s>"

    def getEmissionProb(self, sents, tagset):
        # P(word|tag) = emissionProb[tag].prob(word)
        emission = []
        for s in sents:
            emission += [(w.lower(), t) for (w, t) in s] # treat for both lowercase and uppercase in the same way

        emissionProb = {}
        for tag in tagset:
            words = [w for (w, t) in emission if t == tag]
            emissionProb[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        return emissionProb

    def getTransitionProb(self, sents, tagset):
        # P(nextTag|prevTag) = transitionProb[prevTag].prob(nextTag)
        transition = []
        for s in sents:
            tags = [t for (w, t) in s]
            transition += ngrams(tags,2)

        transitionProb = {}
        for tag in tagset:
            nextTags = [nextTag for (prevTag, nextTag) in transition if prevTag == tag]
            transitionProb[tag] = WittenBellProbDist(FreqDist(nextTags), bins=1e5)

        return transitionProb

    def evaluate(self, sent, pred, comparision):
        # print "word\t\tactual tag\t\tpredited tag"
        for i in range(1, len(sent)-1):
            # print sent[i][0]+"\t\t"+sent[i][1]+"\t\t"+pred[i][1]
            assert (sent[i][0].lower() == pred[i][0])

            if sent[i][1] == self.startWord or sent[i][1] == self.endWord:
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