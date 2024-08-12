# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:57:04 2022

@author: USER
"""

import os
import nltk
import string
import pandas as pd
import numpy as np
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sortedcontainers import SortedList, SortedSet, SortedDict

#make system path
path_spam = os.path.join(os.getcwd(), 'dataset\\spam')
path_ham = os.path.join(os.getcwd(), 'dataset\\ham')
path_test = os.path.join(os.getcwd(), 'test')

#collection of mails
spam = []
ham = []
document = []

def read_text_file(file_path, storage):
    f = open(file_path, errors="ignore")
    storage.append(f.read())

def read_spam_data():
    # Change the directory to read soam data
    os.chdir(path_spam)
    # iterate through all file
    for file in os.listdir():
            read_text_file(file, spam)
            
def read_ham_data():
    # Change the directory to read ham data       
    os.chdir(path_ham)
    # iterate through all file
    for file in os.listdir():
            read_text_file(file, ham)
     
#text cleaning   
def remove_punctuation_and_stopwords(email):
    
    email_no_punctuation = [ch for ch in email if ch not in string.punctuation]
   
    #sms_no_punctuation = "".join(sms_no_punctuation).split()
    email_no_punctuation_no_stopwords = [word.lower() for word in email_no_punctuation if word.lower() not in stopwords.words("english")]
    email_no_punctuation_no_stopwords_isalpha =  [word.lower() for word in email_no_punctuation_no_stopwords if (word.isalpha() == True)]
    return email_no_punctuation_no_stopwords_isalpha

def clean_all_data():
    for i in range(len(spam)):
        nltk_tokens =  nltk.word_tokenize(spam[i])
        spam[i] = remove_punctuation_and_stopwords(nltk_tokens)
    
    for i in range(len(ham)):
        nltk_tokens =  nltk.word_tokenize(ham[i])
        ham[i] = remove_punctuation_and_stopwords(nltk_tokens)

def create_vectorizer():
    #combining all the data in documents
    for i in range(len(spam)):
        document.append(" ".join(spam[i]))
            
    for i in range(len(ham)):
        document.append(" ".join(ham[i])) 
        

read_spam_data()
read_ham_data()
clean_all_data()
create_vectorizer()

#count vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(document)
#print("Vocabulary: ", vectorizer.vocabulary_)
header = vectorizer.vocabulary_.keys()
header = list(header)
vector = vectorizer.transform(document)

X = vector.toarray()


spam_data = 0
ham_data = 0
for i in range(len(spam)):
    spam_data += X[i]

non_spam_data = 0
for i in range(len(ham)):
    non_spam_data += X[i + len(spam)]
    

spam_data = spam_data + np.ones(X.shape[1])
non_spam_data = non_spam_data + np.ones(X.shape[1])
spam_data = spam_data / (len(spam) + 1)
non_spam_data = non_spam_data / (len(ham) + 1)
spam_data = spam_data / sum(spam_data)
non_spam_data = non_spam_data/sum(non_spam_data)


p_ml = len(ham)/(len(spam)+len(ham))
q_ml = 1 - p_ml

#now build the seperator
w = np.zeros(X.shape[1])

for i in range(X.shape[1]):
    nom = spam_data[i]*(1-non_spam_data[i])
    denom = non_spam_data[i] * (1 - spam_data[i])
    w[i] = math.log(nom/denom)

#now build the bias term   
bias = math.log(p_ml/q_ml)
for i in range(X.shape[1]):
    bias += math.log((1 - spam_data[i])/(1 - non_spam_data[i]))  

decision = []

for i in range(X.shape[0]):
    d = np.matmul(w.T, (X[i]/sum(X[i]))) + bias
    if(d > 0):
        decision.append(1)
    else:
        decision.append(0)

y = []
for i in range(len(spam)):
    y.append(1)
for i in range(len(ham)):
    y.append(0)
   
    
count = 0
for i in range(len(decision)):
    if(decision[i] != y[i]):
        count += 1
        
#accuracy in train data
acc = (X.shape[0]-count)/X.shape[0]
print('training accuracy', acc * 100, "%")  


def classifier():
    os.chdir(path_test) 
    test_mail_data = []
    test_mail_filename = []
    
    #read test mails
    for file in os.listdir():
        test_mail_filename.append(file)
        read_text_file(file, test_mail_data)
    
    #cleam test mails
    for i in range(len(test_mail_data)):
            nltk_tokens =  nltk.word_tokenize(test_mail_data[i])
            test_mail_data[i] = remove_punctuation_and_stopwords(nltk_tokens)
            test_mail_data[i] = " ".join(test_mail_data[i])
        
        
    #vectorize test mails
    test_vector = vectorizer.transform(test_mail_data).toarray()
    test_predict = []
    
    
    for i in range(len(test_vector)):
        d = np.matmul(w.T, (test_vector[i]/sum(test_vector[i]))) + bias
        if(d > 0):
            test_predict.append(1)
        else:
            test_predict.append(0)
            
    for i in range(len(test_predict)):
        res = ''
        if(test_predict[i] == 1):
            res = 'spam'
        else: 
            res = 'non-spam'
        print(test_mail_filename[i], 'is predicted as', res)
     
#function to return the classification of emails from test folder
classifier()
