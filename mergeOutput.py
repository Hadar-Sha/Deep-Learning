# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 12:36:08 2018

@author: H
"""


import csv

num= 230

basefolder= 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/'
folder = basefolder+ str(num)+ '/'

outfile= basefolder+ 'Output_users_221-230.csv'

infile= folder + 'Output_user_'+ str(num)+'_out.csv'

outfileNew=[[] for k in range(1000)]


with open(infile, newline='') as csvfile:
     dbreader = csv.reader(csvfile)
     next(dbreader)
     j=0
     for row in dbreader:
         new_row= []
         new_row.append(str(num))
         new_row.extend(row)
         outfileNew[j]= new_row
         j+=1
#     to do 
     
with open(outfile, 'a+', newline='') as csvfile:
    mywriter = csv.writer(csvfile, delimiter=',')
    for outrow in outfileNew:
        mywriter.writerow(outrow)