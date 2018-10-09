# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 13:40:55 2018

@author: H
"""

import csv

num= 230

basefolder= 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/real-users-experiment-results/'
folder = basefolder+ str(num)+ '/'

infile= basefolder+'many-users-for-analysis/'+ 'Output_users_201-210-round.csv'
outfile= basefolder+'many-users-for-analysis/'+ 'Output_users_201-210-round-updated.csv'

db1file= folder+ 'Data_one_color_digit.csv'
db2file= folder+ 'Data_same_bg_dist_digit.csv'
db3file= folder+ 'Data_sim_two_digits.csv'
db4file= folder+ 'Data_rand.csv'

outfileNew=[[] for k in range(10000)]

with open(infile, newline='') as csvfile:
     dbreader = csv.reader(csvfile)
     print ("in reading file")
     j=0
#     a=0
     for inrow in dbreader:
         dbtype= int(inrow[1])
         rowindb= int(inrow[2])
         color= inrow[6:30]
#         a=dbtype
         
         if dbtype==1:
#             print ("in dbtype = 1")
             with open(db1file, newline='') as csvfile:
                 db1reader = csv.reader(csvfile)
                 
                 i=1
                 for db1row in db1reader:
                     db1color= db1row[0:24]
                     if rowindb==i:  # if the 
                         if db1color==color:
                             outfileNew[j]=inrow
                             j+=1
#                         else:
#                             outfileNew[j]=[]
                         break
                     else:
                         i+=1
#                         print (dbtype)
#                         print (i)
                         continue
             
         
         elif dbtype==2:
#             print ("in dbtype = 2")
             with open(db2file, newline='') as csvfile:
                 db2reader = csv.reader(csvfile)
                 
                 i=1
                 for db2row in db2reader:
                     db2color= db2row[0:24]
                     if rowindb==i:  # if the 
                         if db2color==color:
                             outfileNew[j]=inrow
                             j+=1
#                         else:
#                             outfileNew[j]=[]
                         break
                     else:
                         i+=1
#                         print (dbtype)
#                         print (i)
                         continue
                         
#             j+=1
             
         elif dbtype==3:
#             print ("in dbtype = 3")
#             outfileNew[j]=[];
#             j+=1          
             continue
         
         elif dbtype==4:
#             print ("in dbtype = 4")
             with open(db4file, newline='') as csvfile:
                 db4reader = csv.reader(csvfile)
                 
                 i=1
                 for db4row in db4reader:
                     db4color= db4row[0:24]
                     if rowindb==i:  # if the 
                         if db4color==color:
                             outfileNew[j]=inrow
                             j+=1
#                         else:
#                             outfileNew[j]=[]
                         
                         break
                     else:
                         i+=1
#                         print (dbtype)
#                         print (i)
                         continue
                         
#             j+=1

print ("reading done")                     
with open(outfile, 'w', newline='') as csvfile:         
    mywriter= csv.writer(csvfile, delimiter=',')
    for row in outfileNew:
        mywriter.writerow(row)                 
print("writing done")
        