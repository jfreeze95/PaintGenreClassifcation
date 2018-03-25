#!/usr/bin/python

#Author: Jessica Freeze
#Date: March 22, 2018
#README: This file is meant to select from the train_info.csv
#file for the Painter By Numbers dataset on Kaggle those entries
#that have more than c pictures of one genre type.
#The filenames and their genres are then saved into train_files.csv
#for use in dataselection.

import pandas as pd

genres = []

f = open('train_info.csv', 'r')

header = f.readline()

for line in f:
	line = line.strip()
	columns = line.split(',')
	genres.append(columns[4])

df = pd.DataFrame({'genredf':genres})
c =df['genredf'].value_counts()
c = c[c>1000]
print len(c.index)


f.close()


f = open('train_info.csv', 'r')
fnew = open('train_files.csv', 'w+')

header = f.readline()

for line in f:
	line = line.strip()
	columns = line.split(',')
	if columns[4] in c:
		fnew.write('%s,%s\r\n' % (columns[0],columns[4]))

fnew.close()

i=0
fnew = open('genres.csv', 'w+')	
while i < len(c.index):
	fnew.write('%s\r\n' % c.index[i])
	i = i+1

fnew.close()

