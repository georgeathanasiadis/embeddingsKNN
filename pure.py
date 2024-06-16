import networkx as nx
import numpy as np
import time
import graph_methods

def main():
    #load the edgelist into a graph
    data_file = 'files/email-Eu-core.txt'
    num_clusters = 42

    G = nx.read_edgelist(data_file)
    G = graph_methods.knn_graph(G, 10)
    
    #track start time
    start_time = time.time()

    #compute shortest path distances between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    #initialize a distance matrix
    num_nodes = len(G.nodes)
    max_distance = max(max(path.values()) for path in shortest_paths.values())  #Maximum shortest path distance
    distance_matrix = np.zeros((num_nodes, num_nodes))

    #fill the distance matrix with shortest path distances
    for i, node1 in enumerate(G.nodes):
        for j, node2 in enumerate(G.nodes):
            if node2 in shortest_paths[node1]:
                distance_matrix[i][j] = shortest_paths[node1][node2]
            else:
                #assign the maximum distance value for disconnected nodes
                distance_matrix[i][j] = max_distance


    print('Start Clustering')
    #three (3) alternatives offered: Agglomerative, KMeans & MiniBatchKMeans
    clusters = graph_methods.perform_clustering_miniBatchkmeans(distance_matrix, num_clusters)

    print('Finished Clustering')
    print(f"Clustering time: {time.time() - start_time} sec")

    #clusters for n value
    cluster_dict = {}
    for i, node in enumerate(G.nodes()):
        if clusters[i] not in cluster_dict:
            cluster_dict[clusters[i]] = []
        cluster_dict[clusters[i]].append(node)

    graph_methods.print_clusters(clusters, num_clusters)
    #graph_methods.output_clusters(clusters, num_clusters)
    graph_methods.evaluate(G, cluster_dict, start_time, distance_matrix, clusters)

if __name__ == "__main__":
    main()