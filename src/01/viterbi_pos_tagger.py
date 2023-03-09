from pos_tagger import pos_tagger


class viterbi_pos_tagger(pos_tagger):
    tagset = set()

    def set_tagset(self, sents):
        for s in sents:
            t = [t for (w, t) in s]
            self.tagset.update(t)

    def initialse(self, sentence, emission_prob, transition_prob):
        vtb = {}
        for tag in self.tagset:
            vtb[tag] = [None for w in sentence] + [None]

        # initialise
        for tag in self.tagset:
            word = sentence[1]
            p = transition_prob[self.START_OF_SENTENCE_MARKER].prob(tag) * emission_prob[tag].prob(word)
            vtb[tag][1] = (self.START_OF_SENTENCE_MARKER, p)

        return vtb

    def run(self, vtb, sentence, emission_prob, transition_prob):
        for i in range(2, len(sentence) - 1):
            word = sentence[i]
            for tag in self.tagset:
                max_prob = (None, 0)
                for prevTag in self.tagset:
                    p = vtb[prevTag][i - 1][1] * transition_prob[prevTag].prob(tag) * emission_prob[tag].prob(word)
                    if p > max_prob[1]:
                        max_prob = (prevTag, p)

                vtb[tag][i] = max_prob

        return vtb

    def finalise(self, vtb, sentence, emission_prob, transition_prob):
        max_prob = (None, 0)
        n = len(sentence) - 1
        for tag in self.tagset:
            p = vtb[tag][n - 1][1] * transition_prob[tag].prob(self.END_OF_SENTENCE_MARKER)
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

    def apply(self, sentence, emission_prob, transition_prob):
        vtb = self.initialse(sentence, emission_prob, transition_prob)
        vtb = self.run(vtb, sentence, emission_prob, transition_prob)
        maxProb = self.finalise(vtb, sentence, emission_prob, transition_prob)

        if maxProb[0] is None:
            return [(w, None) for w in sentence]

        return self.backtrack(vtb, sentence, maxProb)
