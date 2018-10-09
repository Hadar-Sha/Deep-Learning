# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:17:44 2018

@author: H
"""

from numpy import genfromtxt
import numpy as np
import csv

path= 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/many-users-for-analysis/users_201-230-db2-entropy.csv'
M = genfromtxt(path, delimiter=',')
#file= [[] for k in range(250)]
#
#with open(path, newline='') as csvfile:
#     dbreader = csv.reader(csvfile)
#     j=0
#     for inrow in dbreader:
#         file[j]=inrow
#         j+=1

#E= (-M*np.log2(M)).sum(axis=1)

# to do- a mat of EMD between each 2 digits- the dig with majority and others in the votes