
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pickle
import numpy as np 

def clustering_evaluation(model, labels, data):
    result = " Adjusted Rand Index : "+str(metrics.adjusted_rand_score(labels,model.labels_))
    result += "\n Homogeneity Score : "+str(metrics.homogeneity_score(labels,model.labels_))
    result += "\n Silhoutte Score : "+str(metrics.silhouette_score(data,model.labels_,metric = 'l2'))
    return result,contingency_matrix(labels, model.labels_)


features = pickle.load(open('../../pickle/features.pkl','rb'))
values = pickle.load(open('../../pickle/values.pkl','rb'))
clustering = AgglomerativeClustering(n_clusters = 5,linkage ='complete')
clustering.fit(features)
evaluation,matrix = clustering_evaluation(clustering,values, data=features)
print(evaluation)
print("Contingency Matrix of clustering done - \n")
print(matrix)
svd = TruncatedSVD(n_components=2)
svd.fit(features)
reduced_data = svd.transform(features)
AC = AgglomerativeClustering(n_clusters=5,linkage ='complete')
AC.fit(reduced_data)

def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    for i in range(X_red.shape[0]):
        plt.scatter(X_red[i, 0], X_red[i, 1],
                 color=plt.cm.nipy_spectral(labels[i] / 5.))

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../../plots/agglomerative_average_lexicalfeatures.png')


# In[204]:


plot_clustering(reduced_data, AC.labels_, 'Agglomerative Clustering with complete Linkage. K=5')


