# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:16:39 2018

@author: H
"""

"""
import re
import os

def files():
    n = 0
    while True:
        n += 1
        yield open('C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/xyY-org-vals-del/one_color_digit_calib_p%d.txt' % n, 'w',  encoding="utf8")
        
        
fs = files()
outfile = next(fs) 
filename= 'C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/xyY-org-vals-del/one_color_digit_calib.txt'

with open(filename,  encoding="utf8") as infile:
    for line in infile:
        if line not in ['\n', '\r\n',''] or not line.strip():
            if pat not in line:
                
                outfile.write(line)
            else:
                items = line.split(pat)
                outfile.write(items[0])
                for item in items[1:]:
                    outfile = next(fs)
                    outfile.write(pat + item)
"""                 
                    
from itertools import zip_longest

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

n = 20

with open('C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/xyY-org-vals-del/sim_two_digits_calib.txt') as f:
    for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
        with open('C:/Users/H/Documents/Haifa Univ/Thesis/experiment/experiment-input/xyY-org-vals-del/sim_two_digits_calib_p{0}.txt'.format(i), 'w') as fout:
            fout.writelines(g)