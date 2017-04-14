import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

# # Creating train and test datasets
# def init_process(fin,fout):
# 	outfile = open(fout,'a',encoding='latin-1')
# 	with open(fin,buffering=200000,encoding='latin-1') as f:
# 		try:
# 			for line in f:
# 				line = line.replace('"','')
# 				initial_polarity = line.split(',')[0]
# 				if initial_polarity == '0':
# 					initial_polarity = [1,0]
# 				elif initial_polarity == '4':
# 					initial_polarity = [0,1]

# 				tweet = line.split(',')[-1]
# 				outline = str(initial_polarity)+':::'+tweet
# 				outfile.write(outline)
# 		except Exception as e:
# 			print(str(e))
# 	outfile.close()

# init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
# init_process('testdata.manual.2009.06.14.csv','test_set.csv')


#Creating lexicon
def create_lexicon(fin):
	lexicon = []
	with open(fin,'r',buffering=100000, encoding='latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter +=1
				if (counter/2500.0).is_integer():
					tweet = line.split(':::')[1]
					content += ' ' + tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon+words))
					print(counter,len(lexicon))
		except Exception as e:
			print(str(e))

	with open('lexicon-pickle','wb') as f:
		pickle.dump(lexicon,f)

create_lexicon('train_set.csv')

