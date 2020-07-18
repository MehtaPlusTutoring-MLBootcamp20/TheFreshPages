import skfuzzy as fuzz
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def preprocess(f1):
    f = f1.read()
    f = f.lower().translate(str.maketrans('', '', string.punctuation))
    f = f.translate(str.maketrans('', '', string.digits))
    return f


with open('fedpapersKnown.txt', 'r') as f1:
    f_known = preprocess(f1)
with open('fedpapersContest.txt', 'r') as f2:
    f_contest = preprocess(f2)


def makeDataFrame(f):
    # splits text into paragraphs
    papersL = f.split("\n\n")

    # lists containing the indexes of the start and end line of each paper
    starting_indexes = []
    ending_indexes = []
    for i in range(len(papersL)):
        if papersL[i] == "to the people of the state of new york":
            starting_indexes.append(i)
        if papersL[i] == "publius":
            ending_indexes.append(i)

    authors = []
    paragraphs = []
    for i in range(len(starting_indexes)):
        index = starting_indexes[i] + 1
        while index < ending_indexes[i]:
            authors.append(papersL[starting_indexes[i] - 1])
            paragraphs.append(papersL[index].replace("\n", " "))
            index += 1

    data_dict = {'authors': authors, 'paragraphs': paragraphs}
    df = pd.DataFrame(data=data_dict)
    return df


def filter(f):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(f)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def get_topwords(f):
    filtered_words = filter(f)
    NUM_TOP_WORDS = 10
    all_tokens = word_tokenize(' '.join(filtered_words))
    fdist = nltk.FreqDist(all_tokens)
    return fdist.most_common(NUM_TOP_WORDS)


def filterL(df):
    filtered_list = []
    for i in df.index:
        filtered_list.append(' '.join(filter(df['paragraphs'][i])))
    return filtered_list


df_known = makeDataFrame(f_known)
topwords_known = get_topwords(f_known)
known_nostop = filterL(df_known)

df_contest = makeDataFrame(f_contest)
topwords_contest = get_topwords(f_contest)
contest_nostop = filterL(df_contest)
# print(df_known)
# print(df_contest)
# print(topwords_known)
# print(topwords_contest)
# print(stopwords.words('english'))


xpts = np.zeros(0)  # lexical richness
ypts = np.zeros(0)  # 
labels = np.zeros(1)
for i in df_known.paragraphs:
    words = word_tokenizer.tokenize(i)
    vocab = set(words)
    xpts = np.hstack((xpts, len(vocab) / float(len(words))))

    #fvs_bow = vectorizer.fit_transform(key.split()).toarray().astype(np.float64)
    # normalise by dividing each row by its Euclidean norm
    #fvs_bow /= np.c_[np.apply_along_axis(np.linalg.norm, 1, fvs_bow)]
    #ypts = np.hstack((ypts, fvs_bow))

# Getting trigrams

def get_trigrams(nostop):
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    X1 = vectorizer.fit_transform(nostop)
    features = (vectorizer.get_feature_names())
    #print("\n\nFeatures : \n", features)
    #print("\n\nX1 : \n", X1.toarray())

    # Applying TFIDF
    vectorizer = TfidfVectorizer(ngram_range=(3, 3))
    X2 = vectorizer.fit_transform(known_nostop)
    #scores = (X2.toarray())
    #print("\n\nScores : \n", scores)

    # Getting top ranking features
    sums = X2.sum(axis=0)
    data1 = []
    for col, term in enumerate(features):
        data1.append((term, sums[0, col]))
    ranking = pd.DataFrame(data1, columns=['term', 'rank'])
    words = (ranking.sort_values('rank', ascending=True))
    return words

known_trigrams = get_trigrams(known_nostop)
contest_trigrams = get_trigrams(contest_nostop)
#print(known_trigrams)
#print(contest_trigrams)

## Configure some general styling
sns.set_style("white")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.figsize'] = (8,7)
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
