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
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


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
    documents = []
    authors = []
    for i in range(len(starting_indexes)):
        index = starting_indexes[i] + 1
        doc = []
        while index < ending_indexes[i]:
            doc.append(papersL[index].replace("\n", " "))
            index += 1
        documents.append(' '.join(doc))
        authors.append(papersL[starting_indexes[i] - 1])
    return documents, authors

def process(l):
    # make everything lowercase, remove punctuation and stopwords
    # and only keep words that appear more than once
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

# get weights for each document with TF-IDF and term weights
def tfidf_weights(processed_corpus, term_weights):
    weights = np.zeros(0)
    words_dict = corpora.Dictionary(processed_corpus)
    bow_corpus = [words_dict.doc2bow(text) for text in processed_corpus]
    model = models.TfidfModel(bow_corpus)

    for i in range(len(processed_corpus)):
        tfidf = model[words_dict.doc2bow(processed_corpus[i])]
        total = 0
        for j in range(len(tfidf)):
            total += term_weights[i][j] * tfidf[j][1] / len(processed_corpus[i])
        weights = np.hstack((weights, [total]))
    return np.vstack(weights)

def get_punctuation(corpus):
    punct = np.zeros((len(corpus), 3))
    for i in range(len(corpus)):
        num_words = len(corpus[i].split())
        # Commas per sentence
        punct[i, 0] = corpus[i].count(',') / float(num_words)
        # Semicolons per sentence
        punct[i, 1] = corpus[i].count(';') / float(num_words)
        # Colons per sentence
        punct[i, 2] = corpus[i].count(':') / float(num_words)
    return punct

# converts the list of labels to authors
def num_to_author(labels, authors):
    count = 0
    a1 = 0
    l = []    
    for i in range(len(labels)):
        count = count+1 if labels[i] == 0 else count-1

    if count >= 0: a1 = 1 

    for i in range(len(labels)):
        l.append('Madison') if labels[i] == a1 else l.append('Hamilton')
    return l

# converts the list of authors to labels
def author_to_num(authors, labels):
    count = 0
    a1 = ''
    l = []
    for i in range(len(labels)):
        count = count+1 if authors[i] == 'hamilton' else count-1

    a1 = 'hamilton' if count >= 0 else 'madison'

    for i in range(len(authors)):
        l.append(0) if authors[i] == a1 else l.append(1)
    return l


def get_accuracy(authors, labels):
    count = 0
    for i in range(len(authors)):
        if labels[i] == authors[i]:
            count += 1
    return count / len(labels)


known_corpus, known_authors = makeCorpus(f_known)
processed_known_corpus = process(known_corpus)
known_term_weights = get_term_weights(known_corpus, processed_known_corpus)
known_punct = get_punctuation(known_corpus)

contest_corpus, contest_authors = makeCorpus(f_contest)
processed_contest_corpus = process(contest_corpus)
contest_term_weights = get_term_weights(contest_corpus, processed_contest_corpus)
contest_punct = get_punctuation(contest_corpus)

# training our tfidf models
known_tfidf_weights = tfidf_weights(processed_known_corpus, known_term_weights)
contest_tfidf_weights = tfidf_weights(processed_contest_corpus, contest_term_weights)

# make matrix
known_matrix = np.concatenate((known_tfidf_weights, known_punct), axis=1)
contest_matrix = np.concatenate((contest_tfidf_weights, contest_punct), axis=1)

# train KMeans model
kmeans = KMeans(n_clusters=2, random_state=0).fit(known_matrix)
labels = kmeans.labels_

labels_predict = kmeans.predict(contest_matrix)
authors_predict = num_to_author(kmeans.predict(contest_matrix), known_authors)
accuracy = get_accuracy(author_to_num(known_authors, labels), labels)

print("predicted labels:", labels_predict)
print("predicted authors:", authors_predict)
print("accuracy:", accuracy)
