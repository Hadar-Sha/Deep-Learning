# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:23:47 2018

@author: H
"""

import csv
import os
import glob

maxColorVal= 255


            
folder= 'C:/Users/H/Documents/Haifa Univ/Thesis/deep-learning/deep-input/big-logits/data/'
os.chdir(folder)

for file in os.listdir(folder):
    if file not in glob.glob("*.txt"):
        filename= os.path.splitext(file)[0]
        print (filename)
        infile= folder + file
        outfileNew=[]
        infileLen=0
#        print (infile)
        
        with open(infile, newline='') as csvfile:
            dbreader = csv.reader(csvfile)
            for row in dbreader:
                
                infileLen += 1
                
                temprow= list(map(int, row))
                rowlist= [x/maxColorVal for x in temprow]
                tempDoubleList=['{:.3f}'.format(x) for x in rowlist]
#                print (tempDoubleList)
                outfileNew.append(tempDoubleList)
                
        outfile= folder + filename + '-normalized.csv'
        with open(outfile, 'w+' , newline='') as csvfile:
            mywriter = csv.writer(csvfile, delimiter=',')
            for outrow in outfileNew:
                mywriter.writerow(outrow)
        
        