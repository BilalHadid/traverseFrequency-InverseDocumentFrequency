# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 07:45:40 2019

@author: bilal
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
#with open('fish.txt' , 'r') as f:
#    f_contents=f.read()
#print(f_contents)
corpus=["hello world this is new program","i am not supposed to tell you that","last day to submit assignmnet","i am just kiding"]
vocabulary=set()
for doc in corpus:
    vocabulary.update(doc.split())
print(vocabulary)
vocabulary=list(vocabulary)
word_index={w:id for id , w in enumerate(vocabulary)}
print(word_index)
tfidf=TfidfVectorizer(vocabulary=vocabulary)
tfidf.fit(corpus)
tfidf.transform(corpus)

for doc in corpus:
    score={}
    print(doc)
    print()
    X=tfidf.transform([doc])
    for word in doc.split():
        score[word]=X[0, tfidf.vocabulary_[word]]
    sortedscore=sorted(score.items() , key=operator.itemgetter(1) , reverse=True)
    print(sortedscore)
    print()
    
