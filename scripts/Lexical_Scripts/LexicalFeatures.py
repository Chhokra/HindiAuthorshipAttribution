import os
from HindiTokenizer import Tokenizer
import re
from cltk.tokenize.sentence import TokenizeSentence
import statistics
import pickle 
PATH = '../Data/'
tokenizer = TokenizeSentence('hindi')
files = os.listdir(PATH)
features = []
values = []
for file in files:
	if(os.path.isdir(PATH+file+'/')):
		for inner_file in os.listdir(PATH+file+'/'):
			if(os.path.isdir(PATH+file+'/'+inner_file+'/')):
				for inner_inner_file in os.listdir(PATH+file+'/'+inner_file+'/'):
					values.append(file)
					t = Tokenizer()
					t.read_from_file(PATH+file+'/'+inner_file+'/'+inner_inner_file)
					split_shit = t.generate_sentences()
					final_split_shit = []
					for i in split_shit:
						hello = re.split('\?|\!',i)
						for k in hello:
							final_split_shit.append(k)
					filtered_final_split_shit = []
					for i in final_split_shit:
						if(not(bool(re.match('^\s+$',i)))):
							filtered_final_split_shit.append(i)
					words = []
					for i in filtered_final_split_shit:
						sentence_tokenized = tokenizer.tokenize(i)
						for k in sentence_tokenized:
							words.append(k.strip('\n'))
					length = [len(tokenizer.tokenize(i)) for i in filtered_final_split_shit]
					one = statistics.mean(length)
					two = statistics.stdev(length)
					vocabulary = set(words)
					three = len(vocabulary)/len(words)
					feature = []
					feature = [one,two,three]
					features.append(feature)

file = open("../pickle/features.pkl",'wb')
pickle.dump(features,file)
file1 = open("../pickle/values.pkl",'wb')
pickle.dump(values,file1)
