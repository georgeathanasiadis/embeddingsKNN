from collections import defaultdict

file_path = 'email-Eu-core-department-labels.txt'
#store clusters and their corresponding nodes
cluster_dict = defaultdict(list)

with open(file_path, 'r') as file:
    for line in file:
        if line.strip():  #check if the line is not empty
            node, cluster = map(int, line.split())
            if cluster not in cluster_dict:
                cluster_dict[cluster] = [node]
            else:
                cluster_dict[cluster].append(node)

#convert defaultdict to dictionary
cluster_dict = dict(cluster_dict)

#print the result
for cluster, nodes in sorted(cluster_dict.items()):
    cluster+=1
    print(f"{cluster}: {nodes}")
    #print('Cluster ' f"{cluster}: {nodes}")

print('')