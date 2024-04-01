#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 09:09:01 2021

@author: jonathanelkobi
"""

import json
import docx
import os.path
import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel


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

# Lemmatize the documents.


lemmatizer = WordNetLemmatizer()
datadocs["doc"] = [[lemmatizer.lemmatize(token) for token in doc] for doc in datadocs["doc"]]
print(datadocs["doc"])
counter = 0 
datadocs["doc"] = [[token for token in doc if not token in wordlist] for doc in datadocs["doc"]]

    
    # Create a dictionary representation of the documents.
dictionary = Dictionary(datadocs["doc"])
with open('dictionary.pickle', 'wb') as handle:
    joblib.dump(dictionary, handle)
for freq in [0.8]:
    # Filter out words that occur less than 0 documents, or more than 80% of the documents.
    dictionary.filter_extremes(no_below=50, no_above=freq)
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in datadocs["doc"]]
    # Train LDA model.
    
    
    # Set training parameters.
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None 
    for i in [54,60]:
        num_topics = i
        count = str(i)
        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token
        
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every,
        )
        doclist =[]
        # Per doc model results
        for name in datadocs["docname"]:
            doc = datadocs.loc[datadocs['docname'] == name, "doc"].iloc[0]
            name1 = name.replace("/","")
            doc = dictionary.doc2bow(doc)
            doc_topics = model.get_document_topics(doc)
            print(name)
            print(doc_topics)
            x = str(doc_topics)
            text = x.replace("[", "")
            text = text.replace("]", "")
            text = text.replace("(", "")
            text = text.replace(",", "")
            text = text.replace(")", "")
            lst = text.split(" ")
            text = ",".join(lst)
            docname = count+"freq"+str(freq)
            doclist.append(name +  "," + text)
        joblib.dump(model, 'model%s.jl' %docname)
        f = open("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/Topic Modeling/DocLDAperdoc%sTopics.csv" %docname, "w", encoding = "utf-8")
        f.write("\n".join(doclist))
        # save top topics
        f = open("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/Topic Modeling/DocLDA%sTopics.txt" %docname, "w", encoding = "utf-8")
        for i in model.print_topics(num_topics=int(count)):
            i = str(i)
            f.write("Topic Number" + i + "\n")
        f.close()


