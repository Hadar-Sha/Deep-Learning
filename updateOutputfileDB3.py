# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:59:15 2018

@author: H
"""

import csv
from shutil import copyfile

num= 220

folder = 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/'+str(num)+'/'
infiledb3= folder+'Data_sim_two_digits.csv'

outfileIn=folder+'Output_user_'+str(num)+'_out-partial.csv'
outfileOut=folder+'Output_user_'+str(num)+'_out.csv'

outfileNew=[[] for k in range(10000)]

templist=[]

copyfile(outfileOut, outfileIn)

changedRow=0

with open(outfileIn, newline='') as csvfile:
     dbreader = csv.reader(csvfile)
#     firstrow= dbreader[0]
     next(dbreader)
     j=0
     for row in dbreader:
         subjNum= row[0]
         dbType= row[1]
         rowInDb= row[2]
         OriginalDigit= row[3]
         seendigit= row[4]
         resptime= row[5]
         color= row[5:29]
#         color= row[6:30]
#         aditparams= row[35:37]
         
         with open(infiledb3, newline='') as csvfile:
             db3reader = csv.reader(csvfile)
             i=0
             for row3 in db3reader:
                 color3= row3[0:24]
                 digit= row3[-2]
                 aditparams=row3[24:-2]
#                 tempDist= float(row3[24])
                 dist= str(round(float(row3[24]),2))
#                 print (dist)
                 otherDigit= row3[25]
                 if (color3== color):

                     newRow=[];
                     newRow.extend(row[0:5])
#                     newRow.extend(row[0:6])
                     newRow.extend(color)
                     newRow.extend(['0'] * 5)
                     newRow.append(dist)
                     newRow.extend(otherDigit)
#                     newRow.extend(aditparams)
                     newRow.extend(['0'] * 3)
                     
#                     print(newRow)
                     
#                     newRow=[];
#                     newRow.extend(row[0:35])
#                     newRow.extend(aditparams)
#                     newRow.extend(row[37:])
#                     
#                     templist.append(aditparams)
                     
#                     newRow=['3',str(i+1),digit,seendigit,resptime]
#                     newRow.extend(color)
#                     newRow.extend(['0'] * 5)
#                     newRow.extend(aditparams)
#                     newRow.extend(['0'] * 3)
                     outfileNew[j]= newRow
                     i+=1
                     changedRow+=1
                     break
                 else:
                     outfileNew[j]= row
                     i+=1            
         j+=1          
                     
with open(outfileOut, 'w', newline='') as csvfile:         
    mywriter= csv.writer(csvfile, delimiter=',')
    mywriter.writerow(firstrow)
    for row in outfileNew:
        mywriter.writerow(row)