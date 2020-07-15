#import pandas as pd
import nltk
nltk.download('words')
nltk.download('treebank')
import re

with open('fedpapersKnown.txt','r') as f1:
    f = f1.read()

# wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

# splits text into paragraphs
fList = f.split("\n\n")

# lists containing the indexes of the start and end line of each paper
starting_indexes = []
ending_indexes = []
for i in range(len(fList)):
    if fList[i] == "To the People of the State of New York:":
        starting_indexes.append(i)
    if fList[i] == "PUBLIUS":
        ending_indexes.append(i)
# print(len(starting_indexes)) -> 71
# print(len(ending_indexes)) -> 71

#for i in range(71):
#    print(starting_indexes[i])
#    print(fList[starting_indexes[i] - 1])

# everything above here works!!

# dict to be turned into pandas dataframe
data_dict = {}
for i in range(71): # range(len(starting_indexes)):
    index = starting_indexes[i] + 1
    author = fList[starting_indexes[i] - 1]
    while index < ending_indexes[i]:
        data_dict[fList[index]] = author
        index += 1

# i'm not sure how to check data_dict as dictionaries are unordered :(

# print(len(data_dict), len(fList)) # -> 1013, 1647 (does this make sense?)

# for x in data_dict: # iterates through the dictionary, too much to fit in terminal so i can't check
#    print(x, data_dict.get(x))