
from sklearn.cluster import KMeans
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
clustering = KMeans(n_clusters = 5,init = 'k-means++')
clustering.fit(features)
evaluation,matrix = clustering_evaluation(clustering,values,data=features)
print(evaluation)
print("Contingency Matrix of clustering done - \n")
print(matrix)

svd = TruncatedSVD(n_components=2)
svd.fit(features)
reduced_data = svd.transform(features)
kmeans = KMeans(n_clusters=5)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the  (SVD-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('kmeans_lexicalfeatures.png')
plt.show()
