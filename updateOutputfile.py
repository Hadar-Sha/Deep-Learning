# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:37:46 2018

@author: H
"""

import csv

folder = 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/'
infiledb3= folder+'Data_sim_two_digits.csv'

outfileIn=folder+'Output_users_211-220-bad.csv'
outfileOut=folder+'Output_users_211-220.csv'

#folder = 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/201/'
infiledb1= folder+'one_color_digit.csv'
infiledb2= folder+'same_bg_dist_digit.csv'
infiledb3= folder+'sim_two_digits.csv'
infiledb4= folder+'rand.csv'

#outfileIn=folder+'user_201_out-short.csv'
#outfileOut=folder+'user_201_out.csv'
#db1=[]
#db2=[]
#db3=[]
#db4=[]

outfileNew=[[] for i in range(1000)]

#db1colors=[]
#db2colors=[]
#db3colors=[]
#db4colors=[]
#
#outcolors=[]

with open(outfileIn, newline='') as csvfile:
     dbreader = csv.reader(csvfile)
     next(dbreader)
     j=0
     for row in dbreader:
         color= row[0:24]
         seendigit= row[-2]
         resptime= row[-1]
         
         
         
         with open(infiledb1, newline='') as csvfile:
             db1reader = csv.reader(csvfile)
             i=0
#             firstTimeFlag=0
#             for i in range(len(db1reader)):
#                 row1= db1reader[i]
             for row1 in db1reader:
                 color1= row1[0:24]
                 digit= row1[-2]
                 aditparams=row1[24:-2]
                 if (color1== color): # and firstTimeFlag==0):
#                     firstTimeFlag=1
#                     colorstr=  ', '.join(color)
#                     aditparamsstr= ', '.join(aditparams)
#                     zeros= [0] * 8
#                     zerosstr= ', '.join(str(zeros))
#                     newRow=[1,i,digit,seendigit,resptime,colorstr,aditparamsstr,zerosstr]
                     newRow=['1',str(i+1),digit,seendigit,resptime]
                     newRow.extend(color)
                     newRow.extend(aditparams)
                     newRow.extend(['0'] * 8)
#                     print (outfileNew[i])
#                     outfileNew[j].append(newRow)
                     outfileNew[j]= newRow
                     i+=1
#                     j+=1
                 else:
                     i+=1
#                     j+=1


         with open(infiledb2, newline='') as csvfile:
             db2reader = csv.reader(csvfile)
             i=0
             for row2 in db2reader:
                 color2= row2[0:24]
                 digit= row2[-2]
                 aditparams=row2[24:-2]
                 if (color2== color):
                     newRow=['2',str(i+1),digit,seendigit,resptime]
                     newRow.extend(color)
                     newRow.extend(['0'] * 2)
                     newRow.extend(aditparams)
                     newRow.extend(['0'] * 5)
#                     outfileNew[j].append(newRow)
                     outfileNew[j]= newRow
                     i+=1
                 else:
                     i+=1
                     
         with open(infiledb3, newline='') as csvfile:
             db3reader = csv.reader(csvfile)
             i=0
             for row3 in db3reader:
                 color3= row3[0:24]
                 digit= row3[-2]
                 aditparams=row3[24:-2]
                 if (color3== color):
                     newRow=['3',str(i+1),digit,seendigit,resptime]
                     newRow.extend(color)
                     newRow.extend(['0'] * 5)
                     newRow.extend(aditparams)
                     newRow.extend(['0'] * 3)
#                     outfileNew[j].append(newRow)
                     outfileNew[j]= newRow
                     i+=1
                 else:
                     i+=1            
         
 
                     
         with open(infiledb4, newline='') as csvfile:
             db4reader = csv.reader(csvfile)
             i=0
             for row4 in db4reader:
                 color4= row4[0:24]
                 digit= row4[-2]
                 aditparams=row4[24:-2]
                 if (color4== color):
                     newRow=['4',str(i+1),digit,seendigit,resptime]
                     newRow.extend(color)
                     newRow.extend(['0'] * 7)
                     newRow.extend(aditparams)
                     outfileNew[j]=newRow
                     i+=1
                 else:
                     i+=1            
                     
         j+=1
#         outfile.append(row)
#         outcolors.append(color)

with open(outfileOut, 'w', newline='') as csvfile:         
    mywriter= csv.writer(csvfile, delimiter=',')
    for row in outfileNew:
        mywriter.writerow(row)
#    mywriter.

#         digit= row[-2]
#         db1.append(row)
#         db1colors.append(color)

#with open(infiledb2, newline='') as csvfile:
#     dbreader = csv.reader(csvfile)
#     for row in dbreader:
#         color= row[0:24]
#         digit= row[-2]
#         db2.append(row)
#         db2colors.append(color)
#         
#         
#with open(infiledb3, newline='') as csvfile:
#     dbreader = csv.reader(csvfile)
#     for row in dbreader:
#         color= row[0:24]
#         digit= row[-2]
#         db3.append(row)
#         db3colors.append(color)
#         
#
#with open(infiledb4, newline='') as csvfile:
#     dbreader = csv.reader(csvfile)
#     for row in dbreader:
#         color= row[0:24]
#         digit= row[-2]
#         db4.append(row)
#         db4colors.append(color)
#         
#         
#
#         
#         
#
##print (db1)
#
##db1= csv.reader(infiledb1)
##db2= csvread(infiledb2);
##db3= csvread(infiledb3);
##db4= csvread(infiledb4);