import time
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
import community
simplefilter("ignore", ClusterWarning)

def louvain_method(G, start_time):
    #Louvain
    partition = community.best_partition(G)

    #organize nodes into clusters
    clusters = {}
    for node, community_id in partition.items():
        if community_id not in clusters:
            clusters[community_id] = []
        clusters[community_id].append(node)

    #sort clusters by community ID
    sorted_clusters = dict(sorted(clusters.items()))

    #print the clusters
    for cluster_num, nodes in sorted_clusters.items():
        # Convert node labels to integers
        nodes = [int(node) for node in nodes]
        print(f'{cluster_num + 1}: {nodes}')

    #execution time
    execution_time = time.time() - start_time
    #print("Cluster Count: ", len(clusters))
    print(f"Execution time: {execution_time} seconds")
    optimal_clusters = len(clusters)

    return optimal_clusters

def elbow_method(distance_matrix):
    # calculate the within-cluster sum of squares (wcss) for different values of k
    wcss = []
    max_k = 10  # maximum value of k to consider
    for k in range(2, max_k + 1):
        linkage_matrix = linkage(distance_matrix, method='ward')
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        cluster_centers = [np.mean(distance_matrix[np.where(clusters == i)], axis=0) for i in range(1, k + 1)]
        wcss.append(sum(
            np.sum((distance_matrix[np.where(clusters == i)] - cluster_centers[i - 1]) ** 2) for i in range(1, k + 1)))

    # plot the within-cluster sum of squares (wcss) against the number of clusters (k)
    plt.plot(range(2, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters (Agglomerative)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(range(2, max_k + 1))
    plt.show()

    # find the optimal k value using the elbow method
    optimal_clusters = np.argmin(np.diff(wcss)) + 2  # Adding 2 because of 0-based indexing
    return optimal_clusters

def silhouette_score_method(distance_matrix):
    #determine silhouette scores for different values of n
    silhouette_scores = []
    max_n = 100  # maximum value of k to consider
    for k in range(2, max_n + 1):
        agglomerative = AgglomerativeClustering(n_clusters=k)
        cluster_labels = agglomerative.fit_predict(distance_matrix)
        silhouette_scores.append(silhouette_score(distance_matrix, cluster_labels))

    #plot the silhouette scores
    plt.plot(range(2, max_n + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal n (Agglomerative)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(range(2, max_n + 1))
    plt.show()

    #find the optimal k value
    optimal_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because of 0-based indexing

    #print the optimal k value
    #print(f'Optimal n value: {optimal_n}')

    return optimal_clusters

def main():

    data_file = 'files/email-Eu-core.txt'
    G = nx.read_edgelist(data_file)

    #track start time
    start_time = time.time()

    #shortest path distances between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    #initialize a distance matrix
    num_nodes = len(G.nodes)
    max_distance = max(max(path.values()) for path in shortest_paths.values())  #maximum shortest path distance
    distance_matrix = np.zeros((num_nodes, num_nodes))

    #distance matrix with shortest path distances
    for i, node1 in enumerate(G.nodes):
        for j, node2 in enumerate(G.nodes):
            if node2 in shortest_paths[node1]:
                distance_matrix[i][j] = shortest_paths[node1][node2]
            else:
                #assign the maximum distance value for disconnected nodes
                distance_matrix[i][j] = max_distance

    #Elbow Method result
    #WARNING: not very robust; the elbow method optimal value must be infered more intuitevely based on the plot.
    #Also, see criticism: https://arxiv.org/abs/2212.12189
    #print("Elbow Method Optimal Clusters:", elbow_method(distance_matrix))

    #Silhouette Score method result
    #print("Silhouette Score method optimal clusters:", silhouette_score_method(distance_matrix))

    #Louvain method result
    print("Louvain method optimal clusters:", louvain_method(G, start_time))

if __name__ == "__main__":
    main()