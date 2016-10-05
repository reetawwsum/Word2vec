from __future__ import print_function

import os
import numpy as np
import zipfile
from collections import Counter
import tensorflow as tf

dataset_path = 'dataset/'
dataset = 'text8.zip'

vocabulary_size = 50000

def read_data(filename):
	'''Reading dataset as a list of words'''
	with zipfile.ZipFile(dataset_path + filename) as f:
			data = tf.compat.as_str(f.read(f.namelist()[0])).split()

	return data

words = read_data(dataset)

# print(len(words))
# 17005207

def build_dataset(words):
	'''Creating vocabulary(index) of most common [vocabulary_size] words'''
	# print(len(Counter(words)))
	# 253854
	count = [['UNK', -1]]
	count.extend(Counter(words).most_common(vocabulary_size - 1))

	dictionary = dict()
	data = list()
	unk_count = 0

	for word, _ in count:
		dictionary[word] = len(dictionary)

	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1
		data.append(index)

	count[0][1] = unk_count

	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
