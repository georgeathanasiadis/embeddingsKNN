from math import log
import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score, homogeneity_score, completeness_score, v_measure_score, \
    fowlkes_mallows_score, adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score

import graph_methods


#from read_cluster_fin import read_cluster

def convert_to_common_format(original_clustering):
    common_format = []
    for i, cluster in enumerate(original_clustering):
        for point in cluster:
            common_format.append((point, i))
    common_format.sort()  #sort by point index to ensure the order is preserved
    return np.array([label for _, label in common_format])

def main():
    #define clusterings to compare
    clustering1 = graph_methods.read_cluster('files/ground_truth.txt')
    clustering2 = graph_methods.read_cluster('files/embeddings.txt')

    max_length = max(len(lst) for lst in clustering1)

    #pad shorter lists with a placeholder value (e.g., -1)
    clustering1_padded = [lst + [-1] * (max_length - len(lst)) for lst in clustering1]

    #convert to ndarray
    ndarray = np.array(clustering1_padded)
    #Variation of Information (VI)

    #UNLIKE THE OTHER METRICS: THE LOWER, THE BETTER
    #maximum VI is log(N), (N = node_count) = log(10) = 3.322, minimum is 0
    #as seen in: https://gist.github.com/jwcarr/626cbc80e0006b526688
    def variation_of_information(cluster1, cluster2):
        total_elements = float(sum([len(c) for c in cluster1]))
        information_variation = 0.0

        for subset1 in cluster1:
            p_value = len(subset1) / total_elements

            for subset2 in cluster2:
                q_value = len(subset2) / total_elements
                intersection_size = len(set(subset1) & set(subset2)) / total_elements

                if intersection_size > 0.0:
                    information_variation += intersection_size * (
                        log(intersection_size / p_value, 2) + log(intersection_size / q_value, 2)
                    )

        return abs(information_variation)

    print("Variation of Information (VI): ", variation_of_information(clustering1, clustering2))

    #for the evaluation methods that are to follow to follow, a format conversion is required
    converted_clustering1 = convert_to_common_format(clustering1)
    converted_clustering2 = convert_to_common_format(clustering2)

    # #Adjusted Rand Index (ARI) & Rand Index
    # print("Adjusted Rand Index (ARI): ", adjusted_rand_score(converted_clustering1, converted_clustering2))
    # print("Rand Index (RI): ", rand_score(converted_clustering1, converted_clustering2))
    #
    # #Adjusted Mutual Information (AMI) & Normalized Mutual Information (NMI)
    # print("Adjusted Mutual Information (AMI): ", adjusted_mutual_info_score(converted_clustering1, converted_clustering2))
    # print("Normalized Mutual Information (NMI): ", normalized_mutual_info_score(converted_clustering1, converted_clustering2))
    #
    # #Homogeneity, completeness and V-measure
    # print("Homogeneity: ", homogeneity_score(converted_clustering1, converted_clustering2))
    # print("Completeness:  ", completeness_score(converted_clustering1, converted_clustering2))
    # print("V-Measure: ", v_measure_score(converted_clustering1, converted_clustering2))

    #Variation of Information (VI):
    print(variation_of_information(clustering1, clustering2))

    #Fowlkes-Mallows scores (FMI)
    print(fowlkes_mallows_score(converted_clustering1, converted_clustering2))

    #Adjusted Rand Index (ARI) & Rand Index
    print(adjusted_rand_score(converted_clustering1, converted_clustering2))
    print(rand_score(converted_clustering1, converted_clustering2))

    #Adjusted Mutual Information (AMI) & Normalized Mutual Information (NMI)
    print(adjusted_mutual_info_score(converted_clustering1, converted_clustering2))
    print(normalized_mutual_info_score(converted_clustering1, converted_clustering2))

    #Homogeneity, completeness and V-measure
    print(homogeneity_score(converted_clustering1, converted_clustering2))
    print(completeness_score(converted_clustering1, converted_clustering2))
    print(v_measure_score(converted_clustering1, converted_clustering2))

if __name__ == "__main__":
    main()