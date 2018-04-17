#!/usr/bin/python

from PIL import Image
import re
from resizeimage import resizeimage

f = open('train_files1.csv', 'r')


for line in f:
	line = line.strip()
	line = re.sub('\t','',line)
	columns = line.split(',')
	img = 'train/'+columns[0]
	im = Image.open(img)
	print(columns[0])
	cover = im.resize((100, 100), Image.NEAREST)
	cover.save(img, im.format)
f.close()

f = open('test_files1.csv', 'r')


for line in f:
	line = line.strip()
	line = re.sub('\t','',line)
	columns = line.split(',')
	img = 'test/'+columns[0]
	im = Image.open(img)
	print(columns[0])
	cover = im.resize((100, 100), Image.NEAREST)
	cover.save(img, im.format)
f.close()