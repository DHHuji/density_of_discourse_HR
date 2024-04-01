#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:29:50 2021

@author: jonathanelkobi
"""
# Import required packages

import json
import docx
import os.path
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.stem.wordnet import WordNetLemmatizer

def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    return text

#getText from docx
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)
listdocs = []
lst = []
docs = []
docdic = {}
countdoc = 0
docnames = pd.read_csv("docnames.csv")
#Create dictonary for all docs
for i in range(len(docnames["doc"])):
    name = docnames["doc"].loc[i]
    if os.path.isfile("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/DATABASE/%s.docx" %name) is True:
        s =getText("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/DATABASE/%s.docx" %name)
        docs.append(s)
        countdoc += 1
datadocs = pd.DataFrame.from_dict({"docname":docnames["doc"],"doc":docs})


# country list
df = pd.read_csv("/Users/jonatanalkobi/Desktop/DH/LAB/WCountries.csv")
l = list(df["Short-form name"])
l1 = list(df["GENC 2A Code"])
l2 = list(df["GENC 3A Code"])
df = pd.read_csv("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/Topic Modeling/Countries and nationalities.csv")
l3 = list(df["Name"])
lst = l + l1 +l2 + l3
countries = []
tokenizer = RegexpTokenizer(r'\w+')
for i in lst:
    x = str(i).lower()  # Convert to lowercase.
    x = tokenizer.tokenize(x)
    for c in x:
        countries.append(c)
countries = set(countries)
countries = list(countries)



# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(datadocs["doc"])):
    datadocs["doc"][idx] = str(datadocs["doc"][idx]).lower()  # Convert to lowercase.
    datadocs["doc"][idx] = tokenizer.tokenize(datadocs["doc"][idx]) # Split into words.



# Remove numbers, but not words that contain numbers.
datadocs["doc"] = [[token for token in doc if not token.isnumeric()] for doc in datadocs["doc"]]



#Remove stop words
datadocs["doc"] = [[token for token in doc if not token in set(stopwords.words('english'))] for doc in datadocs["doc"]]



#Remove countries
datadocs["doc"] = [[token for token in doc if not token in countries] for doc in datadocs["doc"]]




# Remove words that are only one character.
datadocs["doc"] = [[token for token in doc if len(token) > 1] for doc in datadocs["doc"]]




wordlist = ["january","february","march","april","may","june","july","august","september","october","november",
            "recommended", "recommendation", "recommends", "recommend","rapporteur","special","commended","unhcr","subcommittee",
            "noted","rom","visit","urge","britian","add","according","asked","december","del","note",
            "often","must","cmw","reservation","question","upon","welcomed","call","covenant","comment","la","un","derechos",
            "en","de","du","le","los","para","cent","crc","ccpr","upr","cerd","cedaw","great","art","state","expert", "independent",
            "committee", "party", "group","palestinian","syrian", "convention","periodic","concerned","mechanism","human",
            "right","government","article","report","inspectorate","call","invite","commends","humanos","droits","viet",
            "libyan","european"]

datadocs["doc"] = [[token for token in doc if not token in wordlist] for doc in datadocs["doc"]]
# Lemmatize the documents.


lemmatizer = WordNetLemmatizer()
datadocs["doc"] = [[lemmatizer.lemmatize(token) for token in doc] for doc in datadocs["doc"]]
print(datadocs["doc"])
counter = 0 
texts = datadocs["doc"]
    
    # Create a dictionary representation of the documents.
dictionary = Dictionary(datadocs["doc"])

freq=0.8
    # Filter out words that occur less than 0 documents, or more than 80% of the documents.
dictionary.filter_extremes(no_below=50, no_above=freq)
    # Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in datadocs["doc"]]
  

    # Train LDA model and get coherence scores

coherence_values = {}
lstnum = [30,40,50,60,70,80]
for num_topics in lstnum:
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values[num_topics] = coherencemodel.get_coherence()



with open('file.txt', 'w') as file:
     file.write(json.dumps(coherence_values))
print(coherence_values)
