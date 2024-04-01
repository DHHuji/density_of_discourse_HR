#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 20:12:39 2021

@author: jonathanelkobi
"""
import pandas as pd
import os

df = pd.read_csv("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/Topic Modeling/DOCSYEARMECH.csv")

f = open("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/Topic Modeling/DocLDAperdoc54freq0.8Topics.csv", "r", encoding = "utf-8")
s = f.read()
s = s.split("\n")
docdic = {}
for i in range(0,60):
    df[i] = 0
counter = 0
for i in s:
    i = i.split(",")
    i[0] = i[0][:-1]
    ii = 2
    for x in i[1::2]:
        x = int(x)
        df[x].loc[counter] =  i[ii]
        print(df[x].loc[counter])
        ii += 2
    counter += 1
    print(i)
df.to_csv("/Users/jonatanalkobi/Desktop/DH/LAB/DATA/Topic Modeling/54Topics.csv")
    
