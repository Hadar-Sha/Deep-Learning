# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:35:28 2018

@author: H
"""

import csv


basefolder= 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/'

infile= basefolder+'many-users-for-analysis/'+ 'Output_users_201-230.csv'
outfile= basefolder+'many-users-for-analysis/'+ 'Output_users_201-230-filtered.csv'

outfileNew=[[] for k in range(30000)]
            
            
with open(infile, newline='') as csvfile:
     dbreader = csv.reader(csvfile)
     print ("in reading file")
     j=0
     for inrow in dbreader:
         RT= int(inrow[5])
         if RT in range(201,1841):
             outfileNew[j]=inrow
             j+=1
             
             
print ("reading done")                     
with open(outfile, 'w', newline='') as csvfile:         
    mywriter= csv.writer(csvfile, delimiter=',')
    for row in outfileNew:
        mywriter.writerow(row)                 
print("writing done") 