# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:56:56 2018

@author: H
"""

import csv

fold= 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/calib-in-RGB/'
outFold='C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/calib-output/'

infileNonRand= fold+'all_data_calib.txt'
infileRand_a= fold+'rand_new_data_calib_a.txt'
infileRand_b= fold+'rand_new_data_calib_b.txt'
infileRand_c= fold+'rand_new_data_calib_c.txt'

miniCalibInfile= fold+'miniCalib500.txt'
nonRandMat=[]
randMat_a=[]
randMat_b=[]
randMat_c=[]

with open(miniCalibInfile, newline='') as csvfile:
        minidbreader = csv.reader(csvfile)
        for row in minidbreader:
            with open(infileNonRand, newline='') as csvfile:
                 dbreader = csv.reader(csvfile)
                 
#                 row= minidbreader[i]
                 for row1 in dbreader:
#                     row1= dbreader[j]
                     if (row== row1):
                         nonRandMat.append(row)
                         
         
     