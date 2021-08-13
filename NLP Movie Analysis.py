#!/usr/bin/env python
# coding: utf-8
pip install nltk
# In[2]:


import nltk
get_ipython().system('python -m pip install -U gensim')
import glob
import multiprocessing
import os
import pprint
import re
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import codecs
import re


# In[3]:


pip install python-Levenshtein


# In[4]:


nltk.download('punkt')#tokenizer, makes words into tokens
nltk.download('stopwords')#like the an, etc


# In[5]:


get_ipython().run_line_magic('pylab', 'inline')


# In[6]:


#get book name
#from glob import glob
import glob
movie_filenames = sorted(glob.glob("/Users/remasbashanfar/Desktop/NLP STUFF/Movie_Database/*json"))
print("here r all the books:")
movie_filenames


# In[7]:


corpus_raw = u""
#for each movie, read it, open it un utf 8 format, 
#add it to the raw corpus
for movie_filename in movie_filenames:
    print("Reading '{0}'...".format(movie_filename))
    with codecs.open(movie_filename, "r", "utf-8") as movie_file:
        corpus_raw += movie_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()


# In[8]:


#load punkt (a trained model) to every word we have
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[9]:


#tokenize into sentences
raw_sentences = tokenizer.tokenize(corpus_raw)


# In[10]:


#convert into word list
#cut off split into words, unecessary characters, no hyhens
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[11]:


sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[12]:


#print
print(raw_sentences[7])
print(sentence_to_wordlist(raw_sentences[7]))


# In[13]:


#count tokens, each one being a sentence
token_counter = sum([len(sentence) for sentence in sentences])
print("The movie corpus has {0:,} tokens".format(token_counter))


# In[14]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[15]:


import gensim.downloader as api


# In[16]:


from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus_raw)


# In[17]:


corpus = api.load('text8')


# In[18]:


import inspect
print(inspect.getsource(corpus.__class__))


# In[19]:


print(inspect.getfile(corpus.__class__))


# In[20]:


from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus)


# In[102]:


print(model.wv.most_similar('killer'))


# In[ ]:





# In[22]:


print(model.wv.most_similar('murderer'))


# In[23]:


print(model.wv.most_similar('ghost'))


# In[24]:


print(model.wv.most_similar('witch'))


# In[25]:


print(model.wv.most_similar('creepy'))


# In[26]:


print(model.wv.most_similar('haunted'))


# In[27]:


import json
info = api.info()
print(json.dumps(info, indent=4))


# In[28]:


print(info.keys())


# In[118]:


for corpus_name, corpus_data in sorted(info['corpora'].items()):
    print(
        '%s (%d records): %s' % (
            corpus_name,
            corpus_data.get('num_records', -1),
            corpus_data['description'][:40] + '...',
        )
    )


# In[119]:


for model_name, model_data in sorted(info['models'].items()):
    print(
        '%s (%d records): %s' % (
            model_name,
            model_data.get('num_records', -1),
            model_data['description'][:40] + '...',
        )
    )


# In[120]:


fake_news_info = api.info('fake-news')
print(json.dumps(fake_news_info, indent=4))


# In[126]:


model = api.load("glove-wiki-gigaword-50")
model.most_similar("creepy") #make counter for each of the similar words and a total counter for all words


# In[127]:


model = api.load("glove-wiki-gigaword-50")
model.most_similar("weird")

#get vector, similar by vector
#model.key to vector , keyedvector


# In[129]:


print(api.load('glove-wiki-gigaword-50'))
model.most_similar("haunted")#vector addition or concat, #add their vectors, #find function that gives vector of a word
#find a way to map them


# In[105]:


print(api.load('glove-wiki-gigaword-50', return_path=True))
#model.most_similar("serial killer")
model.wv.most_similar("serial") 


# In[103]:


print(api.load('glove-wiki-gigaword-50', return_path=True))
#model.most_similar("serial")
model.wv.most_similar("glass") 


# In[107]:


print(api.load('glove-wiki-gigaword-50', return_path=True))
model.wv.most_similar(positive=['woman', 'king'], negative=['man'])


# In[ ]:



print(f"Word 'penalty' appeared {model.wv.get_vecattr('penalty', 'count')} times in the training corpus.")


# In[ ]:


print(f"Word 'killer' appeared {model.wv.get_vecattr('killer', 'count')} times in the training corpus.")


# In[ ]:


print(f"Word 'love' appeared {model.wv.get_vecattr('love', 'count')} times in the training corpus.")


# In[ ]:


print(f"Word 'hate' appeared {model.wv.get_vecattr('hate', 'count')} times in the training corpus.")


# In[ ]:


counter=0
for filepath in glob.iglob(r'/Users/remasbashanfar/Desktop/NLP STUFF/Movie_Database/*.json'):
    file = open(filepath, "rt")
    data = file.read()
    words = data.split()
    counter += len(words)
    print(counter,filepath)


# In[ ]:


words #use a json library in python to read, split words, store token, compare 2 models, how they were trained,

#top 10 similar words, count number of times the keyword 


# In[ ]:


import json


# In[ ]:


#make letters small case, take off !/
#loop through movies, find words, and put a counter
#show top 5 movies
#look up the mid in the file
#look up python dictionary 
#serial killer, 3-5, horror, drama: killer(serial), creepy(child,person) haunted (house, city)
#later, create a map, how many overlap
  counter1=0
for filepath1 in glob.iglob(r'/Users/remasbashanfar/Desktop/NLP STUFF/Movie_Database/*.json'):
    file1 = open(filepath1, "rt")
    data1 = file1.read()
    words1 = data1.split()
    eachWord=[[word.lower() for word in eachWord.split()] for eachWord in words1]
    counter1 += len(eachWord)
    print(counter1,filepath1) 
    
    


# In[114]:


totalCount=0
killerCount=0
killersCount=0
victimCount=0
murdererCount=0
escapesCount=0
killCount=0
mysteriousCount=0
serialCount=0
killsCount=0
newList=[]
for filepath1 in glob.iglob(r'/Users/remasbashanfar/Desktop/NLP STUFF/Movie_Database/*.json'):
    file1 = open(filepath1, "rt")
    data1 = file1.read()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in data1:
        if ele in punc:
            data1 = data1.replace(ele, "")
    words1 = data1.split()
    eachWord=[[word.lower() for word in eachWord.split()] for eachWord in words1]
    file1= eachWord
                
    for i in range(len(file1)):
        if(file1[i] ==['killer']):
            killerCount+=1
            totalCount+=1
        if(file1[i] ==['killers'] or file1[i] =='killers'): 
            killersCount+=1
            totalCount+=1
        if(file1[i] ==['victim']or file1[i] =='victim'): 
            victimCount+=1
            totalCount+=1
        if(file1[i] ==['murderer']or file1[i] =='murderer'): 
            murdererCount+=1
            totalCount+=1
        if(file1[i] ==['escapes'] or file1[i] =='escapes'): 
            escapesCount+=1
            totalCount+=1    
        if(file1[i] ==['kill'] or file1[i] =='kill'):
            killCount+=1
            totalCount+=1
        if(file1[i] ==['kills'] or file1[i] =='kills'): 
            killsCount+=1
            totalCount+=1
        if(file1[i] ==['mysterious'] or file1[i] =='mysterious'): 
            mysteriousCount+=1
            totalCount+=1
        if(file1[i] ==['serial'] or file1[i] =='serial'): 
            serialCount+=1
            totalCount+=1
    newDict=dict({"Movie": file1[1],"totalCount":totalCount,"killersCount": killersCount, "victimCount": victimCount,"murdererCount": murdererCount,"escapesCount": escapesCount, "killCount": killCount,"killsCount":killsCount, "mysteriousCount":mysteriousCount,"serialCount":serialCount})
    newList.append(newDict)
    totalCount=0
    killerCount=0
    killersCount=0
    victimCount=0
    murdererCount=0
    escapesCount=0
    killCount=0
    mysteriousCount=0
    serialCount=0
    killsCount=0
sortedList = sorted(newList, key=lambda x: x['totalCount']) 
print(sortedList)


# In[116]:


file1[100]


# In[ ]:


eachWord 


# In[79]:


similars = model.wv.most_similar('killer')
print(similars)


# In[80]:


similars = model.wv.most_similar('serial')
print(similars)


# In[131]:


totalCount=0
strangeCount=0
creepyCount=0
weirdCount=0
disturbingCount=0
spookyCount=0
alarmingCount=0
sinisterCount=0
scaryCount=0
newList=[]
for filepath1 in glob.iglob(r'/Users/remasbashanfar/Desktop/NLP STUFF/Movie_Database/*.json'):
    file1 = open(filepath1, "rt")
    data1 = file1.read()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in data1:
        if ele in punc:
            data1 = data1.replace(ele, "")
    words1 = data1.split()
    eachWord=[[word.lower() for word in eachWord.split()] for eachWord in words1]
    file1= eachWord
                
    for i in range(len(file1)):
        if(file1[i] ==['creepy']):
            creepyCount+=1
            totalCount+=1
        if(file1[i] ==['strange']): 
            strangeCount+=1
            totalCount+=1
        if(file1[i] ==['weird']): 
            weirdCount+=1
            totalCount+=1
        if(file1[i] ==['disturbing']): 
            disturbingCount+=1
            totalCount+=1
        if(file1[i] ==['spooky']): 
            spookyCount+=1
            totalCount+=1    
        if(file1[i] ==['scary']): 
            scaryCount+=1
            totalCount+=1
        if(file1[i] ==['alarming']): 
            alarmingCount+=1
            totalCount+=1
        if(file1[i] ==['sinister']): 
            sinisterCount+=1
            totalCount+=1
    newDict=dict({"Movie": file1[1],"totalCount":totalCount,"creepy": creepyCount, "strange": strangeCount,"weird": weirdCount,"disturbing": disturbingCount, "spooky": spookyCount, "scary":scaryCount,"alarming":alarmingCount, "sinister":sinisterCount})
    newList.append(newDict)
    totalCount=0
    strangeCount=0
    creepyCount=0
    weirdCount=0
    disturbingCount=0
    spookyCount=0
    alarmingCount=0
    sinisterCount=0
    scaryCount=0
sortedList = sorted(newList, key=lambda x: x['totalCount']) 
print(sortedList)


# In[133]:


totalCount=0
hauntCount=0
ghostCount=0
abandonedCount=0
mysteryCount=0
nightmareCount=0
strangerCount=0
possessedCount=0
exorcismCount=0
sinisterCount=0
newList=[]
for filepath1 in glob.iglob(r'/Users/remasbashanfar/Desktop/NLP STUFF/Movie_Database/*.json'):
    file1 = open(filepath1, "rt")
    data1 = file1.read()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in data1:
        if ele in punc:
            data1 = data1.replace(ele, "")
    words1 = data1.split()
    eachWord=[[word.lower() for word in eachWord.split()] for eachWord in words1]
    file1= eachWord
                
    for i in range(len(file1)):
        if(file1[i] ==['haunts'] or file1[i] ==['haunt'] or file1[i] ==['haunted']):
            hauntCount+=1
            totalCount+=1
        if(file1[i] ==['ghost'] or file1[i] ==['ghosts']): 
            ghostCount+=1
            totalCount+=1
        if(file1[i] ==['abandoned']): 
            abandonedCount+=1
            totalCount+=1
        if(file1[i] ==['mystery']): 
            mysteryCount+=1
            totalCount+=1
        if(file1[i] ==['nightmare']): 
            nightmareCount+=1
            totalCount+=1    
        if(file1[i] ==['stranger']): 
            strangerCount+=1
            totalCount+=1
        if(file1[i] ==['sinister']): 
            sinisterCount+=1
            totalCount+=1
        if(file1[i] ==['possessed']): 
            possessedCount+=1
            totalCount+=1
        if(file1[i] ==['exorcism'] or file1[i] ==['exorcise'] or file1[i] ==['exorcises']): 
            exorcismCount+=1
            totalCount+=1
    newDict=dict({"Movie": file1[1],"totalCount":totalCount,"haunted": hauntCount, "stranger": strangerCount,"ghost": ghostCount,"abandoned": abandonedCount, "mystery": mysteryCount, "nightmare":nightmareCount,"sinister":sinisterCount, "possessed":possessedCount, "exorcism":exorcismCount})
    newList.append(newDict)
    totalCount=0
    hauntCount=0
    ghostCount=0
    abandonedCount=0
    mysteryCount=0
    nightmareCount=0
    strangerCount=0
    possessedCount=0
    exorcismCount=0
    sinisterCount=0
sortedList = sorted(newList, key=lambda x: x['totalCount']) 
print(sortedList)


# In[ ]:




