'''
Created on Mar 2, 2018
'''
from nltk.util import ngrams
from nltk import FreqDist, WittenBellProbDist
from sklearn.linear_model import LinearRegression
import numpy as np
from math import log10

class HMM():
    def __init__(self, train_sents, test_sents):
        self.train_sents = train_sents
        self.test_sents = test_sents
        self.words, self.tags = self.get_words_and_tags()
        self.change_unknown_words()
        self.tags_dist = FreqDist(self.tags)            
        self.words_dist = FreqDist(self.words)
        # if smoothing == "-l":
        self.tag_set = set()
        self.set_tag_set(self.tags)
        print("hoge:",self.tag_set)
        self.transition_prob = self.create_transition_table()
        # else:
        #     self.transition_prob = self.good_turing_transition_table()
        self.emission_prob = self.create_emission_table()
        print("hmm training completed\n")
        print("hoge1:",self.transition_prob)
        print("hoge2:", self.emission_prob)


    def set_tag_set(self, tags):
        # for tag in tags:
        #     print("hoge:", tag)
            # t = [t for (t) in tag]
            self.tag_set.update(tags)


    def get_sentences(self, selected_tagset):
        tagged_sents = self.corpus.tagged_sents(tagset=selected_tagset)
        sents = self.corpus.sents()
        return tagged_sents, sents 
            
    def change_unknown_words(self):           
        words_dist = FreqDist(self.words)
        for index, w in enumerate(self.words):
            if words_dist[w] == 1:
                self.words[index] = "UNK"
        
    def train_test_split(self):
        train_sents = self.tagged_sents[:self.train_size]
        test_sents = self.sents[self.train_size:self.train_size+self.test_size]
        return train_sents, test_sents
    
    def get_words_and_tags(self):
        # create list of words and tags
        words = []
        tags = []
        start = ["<s>"]
        end = ["</s>"]
        for sent in self.train_sents:
            words += start + [w for (w,_) in sent] + end
            tags += start + [t for (_,t) in sent] + end
        return words, tags
    
    def create_transition_table(self):
        # P(nextTag|prevTag) = transitionProb[prevTag].prob(nextTag)
        transition = []
        for s in self.train_sents:
            tags = [t for (w, t) in s]
            transition += ngrams(tags,2)

        transitionProb = {}
        for tag in self.tag_set:
            nextTags = [nextTag for (prevTag, nextTag) in transition if prevTag == tag]
            transitionProb[tag] = WittenBellProbDist(FreqDist(nextTags), bins=1e5)

        return transitionProb
    
    def get_nc(self, c, linreg):
        x = [log10(c)]
        x = np.c_[np.ones_like(x), x]
        y_hat = linreg.predict(x)
        return pow(10, y_hat[0])

    def create_word_tag_pairs(self):
        words_tags = []
        for word, tag in zip(self.words, self.tags):
            words_tags += list(ngrams([tag,word], 2))
        return words_tags
    
    def create_emission_table(self):
        emission = []
        for s in self.train_sents:
            emission += [(w.lower(), t) for (w, t) in s] # treat for both lowercase and uppercase in the same way

        emissionProb = {}
        for tag in self.tag_set:
            words = [w for (w, t) in emission if t == tag]
            emissionProb[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        return emissionProb
