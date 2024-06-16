from collections import deque
#import annchor
import networkx as nx
import nodevectors
from joblib import Parallel, delayed
from karateclub import NodeSketch
from node2vec import Node2Vec
import numpy as np
import time
import graph_methods
import heapq

"""
Embeddings generation methods:
-ProNE
-Node2Vec
-NodeSketch
"""
###################
def generate_embeddings_prone(G, n_components=256, mu=0.5, theta=0.5):
    """
    Generate node embeddings for a graph using the ProNE method.
    See paper: https://www.ijcai.org/proceedings/2019/0594.pdf
    Parameters:
    G : graph
        The input graph for which node embeddings are to be generated.
    n_components : int, optional
        Number of components for the embeddings. Default is 256.
    mu : float, optional
        Parameter for ProNE. Default is 0.5.
    theta : float, optional
        Parameter for ProNE. Default is 0.5.

    Returns:
    embedding_matrix : numpy.ndarray
        The resulting embedding matrix.
    """
    pne = nodevectors.ProNE(
        n_components=n_components,
        mu=mu,
        theta=theta,
    )
    embedding_matrix = pne.fit_transform(G)
    print('Finished Embeddings')
    return embedding_matrix


def generate_embeddings_node2vec(G, dimensions=256, walk_length=60, num_walks=200, workers=4, seed=10):
    """
    Generate node embeddings for a graph using the Node2Vec method.
    See paper: https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf
    Parameters:
    G : graph
        The input graph for which node embeddings are to be generated.
    dimensions : int, optional
        Number of dimensions for the node embeddings. Default is 256.
    walk_length : int, optional
        Length of each random walk.
    num_walks : int, optional
        Number of random walks to perform from each node. Default is 200.
    workers : int, optional
        Number of parallel processes to use for the random walks. Default is 4.
    seed : int, optional
        Random seed for reproducibility. Default is 10.
    window : int, optional
        Window size for word2vec model. Default is 10.
    min_count : int, optional
        Minimum count for word2vec model. Default is 1.
    batch_words : int, optional
        Batch size for word2vec model. Default is 4.

    Returns:
    embedding_matrix : numpy.ndarray
        The resulting embedding matrix.
    """
    start_time = time.time()

    #initialize Node2Vec model
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers,
                        seed=seed)
    #fit the model
    model = node2vec.fit()

    print(f"Embeddings training time: {time.time() - start_time} sec")
    #create the node embeddings dictionary
    node_embeddings = {node: model.wv[node] for node in G.nodes()}

    #create the embedding matrix
    embedding_matrix = np.array([node_embeddings[node] for node in G.nodes()])

    return embedding_matrix


def generate_embeddings_nodesketch(G, dimensions=256):
    """
    Generate node embeddings for a graph using the NodeSketch method.
    See paper: https://exascale.info/assets/pdf/yang2019nodesketch.pdf
    Parameters:
    G : graph
        The input graph for which node embeddings are to be generated.
    dimensions : int, optional
        Number of dimensions for the node embeddings. Default is 256.

    Returns:
    embedding_matrix : numpy.ndarray
        The resulting embedding matrix.
    """
    #preprocess for karateclub embeddings
    G = nx.convert_node_labels_to_integers(G)

    #initialize NodeSketch model
    nodesketch = NodeSketch(dimensions=dimensions)

    #fit the model to the graph
    nodesketch.fit(G)

    #get the embedding matrix
    embedding_matrix = nodesketch.get_embedding()

    return embedding_matrix


def main():
    start_time = time.time()
    seed = 10
    graph_methods.set_seed(seed)
    #np.random.seed(10)
    num_clusters = 42
    data_file = 'files/email-Eu-core.txt'


    G_init = nx.read_edgelist(data_file)
    #G = nx.read_edgelist(data_file)

    #G = knn_graph(G_init, 20)
    G = graph_methods.knn_graph_dynamic(G_init, 0.25)
    #G = nx.convert_node_labels_to_integers(G)

    #draw_graph(G)

    #track start time

    print('Started Embeddings')
    embedding_matrix = generate_embeddings_prone(G)

    # ann = annchor.Annchor(embedding_matrix.tolist(),
    #                       'euclidean',
    #                       n_anchors=15,
    #                       n_neighbors=15,
    #                       p_work=0.1)
    # ann.fit()
    print('Start Clustering')
    #three (3) alternatives offered: Agglomerative, KMeans & MiniBatchKMeans
    clusters = graph_methods.perform_clustering_miniBatchkmeans(embedding_matrix, num_clusters)

    print('Finished Clustering')
    print(f"Clustering time: {time.time() - start_time} sec")

    #clusters for n value
    cluster_dict = {}
    for i, node in enumerate(G.nodes()):
        if clusters[i] not in cluster_dict:
            cluster_dict[clusters[i]] = []
        cluster_dict[clusters[i]].append(node)

    graph_methods.print_clusters(clusters, num_clusters)
    graph_methods.output_clusters(clusters, num_clusters)
    graph_methods.evaluate(G, cluster_dict, start_time, embedding_matrix, clusters)

if __name__ == "__main__":
    main()