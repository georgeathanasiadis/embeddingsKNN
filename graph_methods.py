"""
Clustering methods:
-MiniBatchKMeans
-KMeans
-Hierarchical/Agglomerative
"""
from collections import deque
from datetime import time
import time
import networkx as nx
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from networkx.algorithms.community import modularity
from silhouette import silhouette_score_block



###################
"""
kNN Graph generation methods:
-static k value
-dynamic k value
"""
###################
def knn_graph(G, k):
    knn_graph = nx.Graph()
    for node in G.nodes():
        #compute hop count to all other nodes
        hop_count = nx.single_source_shortest_path_length(G, node)
        #sort nodes by hop count and select k nearest neighbors
        nearest_neighbors = sorted(hop_count, key=hop_count.get)[:k+1] # +1 to include self
        #add edges between the node and its k nearest neighbors
        for neighbor in nearest_neighbors:
            if neighbor != node:
                knn_graph.add_edge(node, neighbor)
            else:
                knn_graph.add_node(node)
    return knn_graph


def knn_graph_scalable(G, k):
    """
    Create a kNN graph from the given unweighted NetworkX graph using an approximate BFS.

    Parameters:
    G (nx.Graph): The input graph.
    k (int): The number of nearest neighbors based on hop count.

    Returns:
    nx.Graph: The kNN graph.
    """
    knn_graph = nx.Graph()

    for node in G.nodes():
        knn_graph.add_node(node) #ensure no nodes are lost, in case they don't yield an edge.
        
    for node in G.nodes():
        bfs_queue = deque([(node, 0)])  #queue of tuples (node, distance)
        distances = {}  #dictionary to store distances

        while bfs_queue and len(distances) < k + 1:  # Include node itself
            current, dist = bfs_queue.popleft()
            if current not in distances:
                distances[current] = dist
                for neighbor in G.neighbors(current):
                    if neighbor not in distances:
                        bfs_queue.append((neighbor, dist + 1))

        #get k-nearest neighbors based on the shortest distances
        neighbors = sorted(distances.items(), key=lambda x: x[1])[:k + 1]

        for neighbor, hop_count in neighbors:
            if neighbor != node:  # Exclude self-loops
                knn_graph.add_edge(node, neighbor)

    return knn_graph


def knn_graph_dynamic(G, percentage):
    knn_graph = nx.Graph()

    for node in G.nodes():
        #compute k as a percentage of the node's degree
        k = int(percentage * G.degree[node]) # int conversion - rounding
        #compute hop count to all other nodes
        hop_count = nx.single_source_shortest_path_length(G, node)
        #sort nodes by hop count and select k nearest neighbors
        nearest_neighbors = sorted(hop_count, key=hop_count.get)[:k+1] # +1 to include self
        #add edges between the node and its k nearest neighbors
        for neighbor in nearest_neighbors:
            if neighbor != node:
                knn_graph.add_edge(node, neighbor)
            else:
                knn_graph.add_node(node)
    return knn_graph


def knn_graph_dynamic_scalable(G, percentage):
    """
    Create a kNN graph from the given NetworkX graph where k is a percentage of each node's degree.
    Since percentage < 1, only the immediate neighbors of each node are traversed for speed & scalability.
    Parameters:
    G (nx.Graph): The input graph.
    k_percent (float): The percentage of each node's degree to use as k.

    Returns:
    nx.Graph: The kNN graph.
    """
    knn_graph = nx.Graph()

    for node in G.nodes():
        knn_graph.add_node(node) #ensure no nodes are lost, in case they don't yield an edge.

    for node in G.nodes():
        degree = G.degree[node]
        k = max(1, int(percentage * degree))  #calculate k as a percentage of the node's degree

        neighbors = sorted(G.neighbors(node), key=lambda x: G.degree(x), reverse=True)[:k]

        for neighbor in neighbors:
            if neighbor != node:  #exclude self-loops
                knn_graph.add_edge(node, neighbor)

    return knn_graph

###################
def read_cluster(fileName):
    with open(fileName, 'r') as file:
        content = file.readlines()

    #process each line and create the list
    clustering = []
    for line in content:
        #find the index of "[" and trim until that point
        start_index = line.index("[")
        processed_line = line[start_index:].replace("]", "],").rstrip(",\n")

        #evaluate the processed line as a list and append to clustering
        clustering.append(eval(processed_line))

    return clustering

def perform_clustering_miniBatchkmeans(embedding_matrix, num_clusters, batch_size=1024, random_state=0, n_init="auto"):
    """
    Perform KMeans clustering on the embedding matrix.

    Parameters:
    embedding_matrix : numpy.ndarray
        The embedding matrix where each row corresponds to the embedding of a node.
    num_clusters : int
        The number of clusters to form.
    batch_size : int, optional
        Size of the mini-batches for the MiniBatchKMeans. Default is 1024.
    random_state : int, optional
        Random seed for reproducibility. Default is 0.
    n_init : str or int, optional
        Number of initializations to perform. Default is "auto".

    Returns:
    clusters : numpy.ndarray
        The resulting cluster labels for each node.
    """
    #initialize MiniBatchKMeans model
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=random_state, batch_size=batch_size, n_init=n_init)
    #fit the model and predict clusters
    clusters = kmeans.fit_predict(embedding_matrix)

    return clusters


def perform_clustering_kmeans(embedding_matrix, num_clusters, random_state=10):
    """
    Perform KMeans clustering on the embedding matrix.

    Parameters:
    embedding_matrix : numpy.ndarray
        The embedding matrix where each row corresponds to the embedding of a node.
    num_clusters : int
        The number of clusters to form.
    random_state : int, optional
        Random seed for reproducibility. Default is 10.

    Returns:
    clusters : numpy.ndarray
        The resulting cluster labels for each node.
    """
    #initialize KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)

    #fit the model and predict clusters
    clusters = kmeans.fit_predict(embedding_matrix)

    return clusters


def perform_clustering_hierarchical(embedding_matrix, num_clusters):
    """
    Perform hierarchical clustering on the embedding matrix using AgglomerativeClustering.

    Parameters:
    embedding_matrix : numpy.ndarray
        The embedding matrix where each row corresponds to the embedding of a node.
    num_clusters : int
        The number of clusters to form.

    Returns:
    clusters : numpy.ndarray
        The resulting cluster labels for each node.
    """
    #perform hierarchical clustering using AgglomerativeClustering
    agglomerative_clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')

    #fit the model and predict clusters
    clusters = agglomerative_clustering.fit_predict(embedding_matrix)

    return clusters

def perform_clustering_hierarchical_visualization(embedding_matrix, num_clusters, method='ward', criterion='maxclust'):
    """
    Perform hierarchical clustering on the embedding matrix and visualize the results.

    Parameters:
    embedding_matrix : numpy.ndarray
        The embedding matrix where each row corresponds to the embedding of a node.
    num_clusters : int
        The number of clusters to form.
    method : str, optional
        The linkage algorithm to use. Default is 'ward'.
    criterion : str, optional
        The criterion to use in forming clusters. Default is 'maxclust'.

    Returns:
    clusters : numpy.ndarray
        The resulting cluster labels for each node.
    """
    #perform hierarchical clustering
    #the linkage function performs hierarchical/agglomerative clustering on the embedding matrix.
    linkage_matrix = linkage(embedding_matrix, method=method)

    #cut the dendrogram to get clusters
    #the fcluster function forms flat clusters from the hierarchical clustering defined by the linkage matrix.
    clusters = fcluster(linkage_matrix, num_clusters, criterion=criterion)

    #plot the dendrogram
    #the dendrogram function generates the plot for the hierarchical clustering.
    plt.figure(figsize=(10, 7))  # Optionally set figure size for better visibility
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram with Node Embeddings')
    plt.xlabel('Graph Nodes')
    plt.ylabel('Distance')
    plt.show()

    return clusters


def draw_graph(G):
    """
    Draw the graph G using NetworkX and Matplotlib.

    Parameters:
    G : networkx.Graph
        The input graph to be drawn.
    """
    #the spring layout positions nodes using Fruchterman-Reingold force-directed algorithm
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)
    nx.draw_networkx_edges(G, pos, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    #display the graph
    plt.title('Graph Visualization')
    plt.show()


def print_clusters(clusters, num_clusters):
    #print the clusters on console
    for cluster_num in range(num_clusters):
        #print(f'Cluster {cluster_num}: {list(np.where(clusters == cluster_num)[0])}')
        print(f'{cluster_num + 1}: {list(np.where(clusters == cluster_num)[0])}')

def output_clusters(clusters, num_clusters):
    #output the clusters to a file
    with open("embeddings_print.txt", "w") as output_file:
        for cluster_num in range(num_clusters):
            print(f'{cluster_num + 1}: {list(np.where(clusters == cluster_num)[0])}')
            nodes_in_cluster = list(np.where(clusters == cluster_num)[0])
            output_file.write(f'{cluster_num}: {nodes_in_cluster}\n')

def set_seed(seed):
    #random.seed(seed)
    np.random.seed(seed)

def evaluate(G, cluster_dict, start_time, matrix, clusters):
    print("Start Density & Coeff")
    #calculate clustering coefficient and density for each cluster individually
    avg_clustering_coefficients = []
    avg_cluster_densities = []
    for cluster_id, nodes in cluster_dict.items():
        subgraph = G.subgraph(nodes)
        avg_clustering_coefficients.append(nx.average_clustering(subgraph))
        avg_cluster_densities.append(nx.density(subgraph))

    #calculate average clustering coefficient and density for all clusters together
    avg_clustering_coefficient = sum(avg_clustering_coefficients) / len(avg_clustering_coefficients)
    avg_cluster_density = sum(avg_cluster_densities) / len(avg_cluster_densities)
    print("Finished Density & Coeff")
    #calculate modularity
    print("Start Modularity")
    modularity_score = modularity(G, [set(nodes) for nodes in cluster_dict.values()])
    print("Finished Modularity")
    print(f"Modularity time: {time.time() - start_time} sec")
    print("Start Int. Measures")
    #internal/intrinsic evaluation measures
    silhouette = silhouette_score_block(matrix, clusters) #more scalable, less robust
    print(f"silhouette time: {time.time() - start_time} sec")
    #silhouette = silhouette_score(matrix, clusters)  #more robust, less scalable
    #print(f"silhouette_score time: {time.time() - start_time} sec")
    calinski_harabasz = calinski_harabasz_score(matrix, clusters)
    print(f"calinski_harabasz time: {time.time() - start_time} sec")
    davies_bouldin = davies_bouldin_score(matrix, clusters)
    print(f"davies_bouldin time: {time.time() - start_time} sec")

    print("Finished Int. Measures")

    #print the scores
    print(f"{time.time() - start_time} sec")
    print(f"{silhouette}")
    print(f"{calinski_harabasz}")
    print(f"{davies_bouldin}")
    print(f"{avg_clustering_coefficient}")
    print(f"{avg_cluster_density}")
    print(f"{modularity_score}")