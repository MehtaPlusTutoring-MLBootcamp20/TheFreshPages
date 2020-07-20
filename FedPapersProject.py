import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pprint
from gensim import corpora
from gensim import models
from collections import defaultdict


with open('fedpapersKnown.txt', 'r') as f1:
    f_known = f1.read().lower()
with open('fedpapersContest.txt', 'r') as f2:
    f_contest = f2.read().lower()

def makeCorpus(f):
    # splits text into paragraphs
    papersL = f.split("\n\n")

    # lists containing the indexes of the start and end line of each paper
    starting_indexes = []
    ending_indexes = []
    for i in range(len(papersL)):
        if papersL[i] == "to the people of the state of new york:":
            starting_indexes.append(i)
        if papersL[i] == "publius":
            ending_indexes.append(i)

    # prepares each paragraph
    paragraphs = []
    for i in range(len(starting_indexes)):
        index = starting_indexes[i] + 1
        while index < ending_indexes[i]:
            paragraphs.append(papersL[index].replace("\n", " "))
            index += 1
    return paragraphs

def process(l):
    # we made everything lowercase, removed punctuation and stopwords
    # and only kept words that appeared more than once
    stop_words = set(stopwords.words('english'))
    processed_list = []
    for i in l:
        i = i.translate(str.maketrans('', '', string.punctuation))
        word_tokens = word_tokenize(i)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        processed_list.append(filtered_sentence)

    frequency = defaultdict(int)
    for text in processed_list:
        for token in text:
            frequency[token] += 1
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in processed_list]
    return processed_corpus

def get_term_weights(corpus, processed_corpus):
    term_weights = []
    for i in range(len(corpus)):
        paragraph = corpus[i]
        word_count = []
        for j in range(len(processed_corpus[i])):
            word_count.append(paragraph.count(processed_corpus[i][j]))
        term_weights.append(word_count)
    return term_weights

def tdidf_weights(processed_corpus, term_weights):
    weights = []
    words_dict = corpora.Dictionary(processed_corpus)
    bow_corpus = [words_dict.doc2bow(text) for text in processed_corpus]
    model = models.TfidfModel(bow_corpus)

    for i in range(len(processed_corpus)):
        tdidf = model[words_dict.doc2bow(processed_corpus[i])]
        sum = 0
        for j in range(len(tdidf)):
            sum += term_weights[i][j] * tdidf[j][1]
        weights.append(sum)
    return weights

def get_pos(corpus):
    pos_list = []
    for i in range(len(corpus)):
        pos = []
        for j in corpus[i]:
            

    # count frequencies for common POS types
    pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']
    fvs_syntax = np.array([[ch.count(pos) for pos in pos_list]
                       for ch in chapters_pos]).astype(np.float64)
 
    # normalise by dividing each row by number of tokens in the chapter
    fvs_syntax /= np.c_[np.array([len(ch) for ch in chapters_pos])]

known_corpus = makeCorpus(f_known)
processed_known_corpus = process(known_corpus)
known_term_weights = get_term_weights(known_corpus, processed_known_corpus)
#known_weights = 

contest_corpus = makeCorpus(f_contest)
processed_contest_corpus = process(contest_corpus)
contest_term_weights = get_term_weights(contest_corpus, processed_contest_corpus)

# training our tdidf models
known_tdidf_weights = tdidf_weights(processed_known_corpus, known_term_weights)
contest_tdidf_weights = tdidf_weights(processed_contest_corpus, contest_term_weights)

