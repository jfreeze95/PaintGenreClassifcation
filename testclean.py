#!/usr/bin/python

#Author: Jessica Freeze
#Date: March 22, 2018
#README: This file is meant to select from the all_data_info.csv
#file for the Painter By Numbers dataset on Kaggle those entries
#that have a genre matching those selected by dataclean.py
# and save their filename and genre to the file test_files.csv

import pandas as pd

genres = []

f = open('genres.csv', 'r')


for line in f:
	line = line.strip()
	columns = line.split(',')
	genres.append(columns[0])

f.close()

ftest = open('all_data_info.csv', 'r')
fdata = open('test_files.csv', 'w+')

for line in ftest:
	line = line.strip()
	columns = line.split(',')
	if columns[2] in genres:
		if columns[10]=='False':
			fdata.write('%s,%s\r\n' % (columns[11],columns[2]))



ftest.close()
fdata.close()

