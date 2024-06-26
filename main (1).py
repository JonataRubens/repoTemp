import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#distancia euclidiana
def dissimilarity(a, b):
    return np.linalg.norm(a - b)

def bsas(data, Q_max, theta):
    clusters = []
    centers = []

    for point in data:
        if len(clusters) == 0:
            clusters.append([point])
            centers.append(point)
        else:
            distances = [dissimilarity(point, center) for center in centers]
            min_distance = min(distances)
            if min_distance > theta and len(clusters) < Q_max:
                clusters.append([point])
                centers.append(point)
            else:
                index = distances.index(min_distance)
                clusters[index].append(point)
                centers[index] = np.mean(clusters[index], axis=0)

    return clusters, centers

#dados IRIS
iris = load_iris()
data = iris.data  

scaler = StandardScaler()
data = scaler.fit_transform(data)

Q_max = 3   # Número máximo de clusters
theta = 1.0 # Limiar de dissimilaridade (ajuste conforme necessário)

clusters, centers = bsas(data, Q_max, theta)
print("\nResultado do Bsas:")
print("Número de clusters formados:", len(clusters))
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {len(cluster)} elementos")

print("\nCentros dos clusters:")
for i, center in enumerate(centers):
    print(f"Centro do Cluster {i + 1}: {center}")




#plot

#pca = PCA(n_components=2)
#data_2d = pca.fit_transform(data)

#plt.figure(figsize=(10, 7))
#colors = ['r', 'g', 'b', 'y', 'c', 'm']

#for i, cluster in enumerate(clusters):
#    cluster_2d = pca.transform(cluster)
#    plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], color=colors[i % len(colors)], label=f'Cluster {i + 1}')

#plt.title('Clusters formados pelo BSAS')
#plt.xlabel('Componente Principal 1')
#plt.ylabel('Componente Principal 2')
#plt.legend()
#plt.show()


########################### K-means ##################################### 

kmeans = KMeans(n_clusters=Q_max, random_state=0)
kmeans.fit(data)


kmeans_clusters = kmeans.labels_
kmeans_centers = kmeans.cluster_centers_

print("\n\n\nResultado do K-means:")
print("Número de clusters formados:", len(np.unique(kmeans_clusters)))
for i in range(Q_max):
    count = np.sum(kmeans_clusters == i)
    print(f"Cluster {i + 1}: {count} elementos")
print("\nCentros dos clusters:")
for i, center in enumerate(kmeans_centers):
    print(f"Centro do Cluster {i + 1}: {center}")

#plot

#pca = PCA(n_components=2)
#data_2d = pca.fit_transform(data)

#plt.figure(figsize=(10, 7))
#colors = ['r', 'g', 'b', 'y', 'c', 'm']

#for i in range(Q_max):
#    cluster_2d = data_2d[kmeans_clusters == i]
#    plt.scatter(cluster_2d[:, 0], cluster_2d[:, 1], color=colors[i], label=f'Cluster {i + 1}')

#plt.title('Clusters formados pelo K-means')
#plt.xlabel('Componente Principal 1')
#plt.ylabel('Componente Principal 2')
#plt.legend()
#plt.show()


#Método do Cotovelo para o K-means
#inertia = []

#for k in range(1, 10):
#    kmeans = KMeans(n_clusters=k, random_state=0)
#    kmeans.fit(data)
#    inertia.append(kmeans.inertia_)

 #Testar diferentes números de clusters

 #Plotar a curva do Método do Cotovelo
#plt.figure(figsize=(8, 6))
#plt.plot(range(1, 10), inertia, marker='o')
#plt.xlabel('Número de clusters')
#plt.ylabel('Inércia')
#plt.title('Método do Cotovelo para Determinação do Número de #Clusters')
#plt.grid(True)
#plt.show()

########################### K-means ####################################


#Método do Cotovelo para o BSAS
#def calculate_bsas_dissimilarity(data, Q_max, theta):
#    dissimilarities = []
#    for k in range(1, Q_max + 1):
#        clusters, centers = bsas(data, k, theta)
#        total_dissimilarity = sum([sum([dissimilarity(point, centers[i]) for point in cluster]) for i, cluster in enumerate(clusters)])
#        dissimilarities.append(total_dissimilarity)
#    return dissimilarities

#bsas_dissimilarities = calculate_bsas_dissimilarity(data, Q_max, theta)

#plt.figure(figsize=(8, 6))
#plt.plot(range(1, Q_max + 1), bsas_dissimilarities, marker='o')
#plt.xlabel('Número de clusters')
#plt.ylabel('Dissimilaridade Total')
#plt.title('Método do Cotovelo para BSAS')
#plt.grid(True)
#plt.show()



















