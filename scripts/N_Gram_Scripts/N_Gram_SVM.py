import numpy as np 
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
values,features = pickle.load(open('../../pickle/n_grams/4grams.pkl','rb'))
features_train,features_test,values_train,values_test = train_test_split(features,values, test_size=0.2)
clf = SVC(gamma='auto')
clf.fit(features_train,values_train)
predictions = clf.predict(features_test)
testing = accuracy_score(values_test,predictions)
precision = precision_score(values_test,predictions,average = 'macro')
recall = recall_score(values_test,predictions,average = 'macro')
print("Multiclass accuracy - " +str(100*testing))
print("Multiclass precision - "+str(precision))
print("Multiclass recall - "+str(recall))

dictionary = {'0':'premchand','1':'sharatchandra','3':'bhairav','4':'vibhooti','2':'dharamveer'}
 

for i in range(0,5):
	values_for_0 = []
	for j in values:
		if(j==dictionary[str(i)]):
			values_for_0.append(j)
		else:
			values_for_0.append("Other")

	features_train,features_test,values_train,values_test = train_test_split(features,values_for_0, test_size=0.2)
	clf = SVC(gamma='auto')
	clf.fit(features_train,values_train)
	predictions = clf.predict(features_test)
	testing = accuracy_score(values_test,predictions)
	precision = precision_score(values_test,predictions,average = 'macro')
	recall = recall_score(values_test,predictions,average = 'macro')
	print("Precision for one vs all for author "+dictionary[str(i)]+" "+str(precision))
	print("Recall for one vs all for author "+dictionary[str(i)]+" "+str(recall))
	print("Accuracy for one vs all for author "+dictionary[str(i)]+" "+str(100*testing))