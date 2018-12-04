import numpy as np 
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statistics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features = pickle.load(open('../../pickle/features.pkl','rb'))
values = pickle.load(open('../../pickle/values.pkl','rb'))
list_accuracies = []
precisions = []
print("Accuracies for multi-class classification - ")

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


dictionary = {'0':'Premchand','1':'Sharatchandra','3':'Bhairav','4':'Vibhuti','2':'Dharamveer'}

for i in range(0,5):
	list_accuracies = []
	values_for_0 = []
	for j in values:
		if(j==str(i)):
			values_for_0.append('0')
		else:
			values_for_0.append('1')
	features_train,features_test,values_train,values_test = train_test_split(features,values_for_0, test_size=0.2)
	clf = SVC(gamma='auto')
	clf.fit(features_train,values_train)
	predictions = clf.predict(features_test)
	testing = accuracy_score(values_test,predictions)
	print("Accuracy for one vs all for author "+dictionary[str(i)]+" "+str(100*testing))
	precision = precision_score(values_test,predictions,average = 'macro')
	recall = recall_score(values_test,predictions,average = 'macro')
	print("Precision for one vs all for author "+dictionary[str(i)]+" "+str(precision))
	print("Recall for one vs all for author "+dictionary[str(i)]+" "+str(recall))
	
	

