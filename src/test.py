def forward_viterbi(y, X, sp, tp, ep):
    T = {}
    for state in X:
        ##          prob.      V. path  V. prob.
        T[state] = (sp[state], [state], sp[state])
    for output in y:
        U = {}
        for next_state in X:
            total = 0
            argmax = None
            valmax = 0
            for source_state in X:
                (prob, v_path, v_prob) = T[source_state]
                p = ep[source_state][output] * tp[source_state][next_state]
                prob *= p
                v_prob *= p
                total += prob
                if v_prob > valmax:
                    argmax = v_path + [next_state]
                    valmax = v_prob
            U[next_state] = (total, argmax, valmax)
        T = U
    ## apply sum/max to the final states:
    total = 0
    argmax = None
    valmax = 0
    for state in X:
        (prob, v_path, v_prob) = T[state]
        total += prob
        if v_prob > valmax:
            argmax = v_path
            valmax = v_prob
    return (total, argmax, valmax)


states = ('Rainy', 'Sunny', 'Cloudy')

observations = ('sleep', 'game', 'eat')

start_probability = {'Rainy': 0.3, 'Sunny': 0.4, 'Cloudy': 0.3}

transition_probability = {
    'Rainy': {'Rainy': 0.4, 'Sunny': 0.3, 'Cloudy': 0.3},
    'Sunny': {'Rainy': 0.2, 'Sunny': 0.7, 'Cloudy': 0.1},
    'Cloudy': {'Rainy': 0.4, 'Sunny': 0.1, 'Cloudy': 0.5}
}

emission_probability = {
    'Rainy': {'sleep': 0.4, 'game': 0.4, 'eat': 0.1},
    'Sunny': {'sleep': 0.2, 'game': 0.7, 'eat': 0.1},
    'Cloudy': {'sleep': 0.2, 'game': 0.2, 'eat': 0.6},
}

total, argmax, valmax = forward_viterbi(observations, states, start_probability, transition_probability, emission_probability)
print("hoge1:", total)
print("hoge2:", argmax)
print("hoge3:", valmax)

def viterbi(observs,states,sp,tp,ep):
    """viterbi algorithm
    Output : labels estimated"""
    T = {} # present state
    for st in states:
        T[st] = (sp[st]*ep[st][observs[0]],[st])
    for ob in observs[1:]:
        T = next_state(ob,states,T,tp,ep)
    prob,labels = max([T[st] for st in T])
    return prob,labels


def next_state(ob,states,T,tp,ep):
    """calculate a next state's probability, and get a next path"""
    U = {} # next state
    for next_s in states:
        U[next_s] = (0,[])
        for now_s in states:
            p = T[now_s][0] * tp[now_s][next_s] * ep[next_s][ob]
            if p>U[next_s][0]:
                U[next_s] = [p,T[now_s][1]+[next_s]]
    return U


prob, labels = viterbi(observations, states, start_probability, transition_probability, emission_probability)
print("hoge1:", prob)
print("hoge2:", labels)

# The number of each tag
# n_tags = Counter(token['upos'] for sent in train_sents for token in sent)
# n_words = Counter(token['form'] for sent in train_sents for token in sent)
# print(n_tags)
# print(n_words)

# tag_fd = nltk.FreqDist(token['upos'] for sent in train_sents for token in sent)
# print("tag_fd = ", tag_fd.most_common(), end = '')
# tag_fd.plot(cumulative = True);

# word_tag_pairs = nltk.bigrams(train_sents)
# noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
# fdist = nltk.FreqDist(noun_preceders)
# print([tag for (tag, _) in fdist.most_common()], end = '')
