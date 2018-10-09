# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:26:33 2018

@author: H
"""

import re
import os

file_list= []
for f in os.listdir('C:\\Users\\H\\Documents\\Haifa Univ\\Thesis\\experiment\\experiment-input\\xyY-org-vals'):
    if f.endswith(".txt"):
        file_list.append(f)

for fileP in file_list:
    in_path= 'C:\\Users\\H\\Documents\\Haifa Univ\\Thesis\\experiment\\experiment-input\\xyY-org-vals\\'+fileP
    out_path='C:\\Users\\H\\Documents\\Haifa Univ\\Thesis\\experiment\\experiment-input\\xyY-org-vals-del\\'+fileP

    file= open(in_path,encoding='utf8')
    o_file= open(out_path,mode='x',encoding='utf8')
    for line in file:
        new_line= re.sub('\n',';\n',line)
        o_file.write(new_line)
        
    o_file.close()    
    file.close()