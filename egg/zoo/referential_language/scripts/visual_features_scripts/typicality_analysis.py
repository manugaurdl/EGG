#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:40:05 2021

@author: eleonora
"""

import numpy as np
import os
import pickle
import statistics

typicality_data_path = ""
files = os.listdir(typicality_data_path)

typicalities = []
for i in files:
    typicalities.append([i.split(".")[0], 
                         pickle.load(open(typicality_data_path + "/" + i, "rb"))])
    
info = []
for i in typicalities:
    info.append([i[0], statistics.mean(i[1]), \
                 statistics.variance(i[1]), 
                 statistics.stdev(i[1])])
    

# STATISTICAL ANALYSES

import pandas as pd
df = pd.DataFrame(info, columns = ["category", "mean typicality", "variance", "stdev"])
df.sort_values(by = "variance", ascending = False)
df.head()

#df.loc[df['category'] == "man"]
