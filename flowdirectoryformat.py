#!/usr/bin/python

#This file will move the selected files to new folders based on their genres so that
#they are in the correct format for the keras flow_from_directory

from shutil import copyfile
import re

#f = open('train_files.csv', 'r')
f = open('train_files_test.csv', 'r')


for line in f:
	line = line.strip()
	line = re.sub('\t','',line)
	columns = line.split(',')
	#src = 'train/'+columns[0]
	#dst = 'train/' + columns[1]+'/'+columns[0]
	src = 'train/'+columns[0]
	dst = 'train/tests/' + columns[1]+'/'+columns[0]
	print (src)
	print(dst)
	copyfile(src, dst)

f.close()

#f = open('test_files.csv', 'r')
f = open('test_files_test.csv', 'r')


for line in f:
	line = line.strip()
	line = re.sub('\t','',line)
	columns = line.split(',')
	#src = 'test/'+columns[0]
	#dst = 'test/' + columns[1]+'/'+columns[0]
	src = 'test/'+columns[0]
	dst = 'test/tests/' + columns[1]+'/'+columns[0]
	print (src)
	print(dst)
	copyfile(src, dst)

f.close()