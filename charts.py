import matplotlib.pyplot as plt

def barchart(x_values, y_values, x_label, y_label, title):
    #create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, color='skyblue')

    #labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x_values)
    plt.ylim(100, 140)

    plt.show()

def linechart(x_values, y_values, x_label, y_label, title):

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', color='skyblue', linestyle='-')

    #labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x_values)
    plt.ylim(100, 140)

    plt.show()

def main():
    x = [2, 5, 10, 20, 30]
    x_lbl = 'k value'

    #Execution time
    y = [131.1303973197937, 112.84199, 114.57211, 130.83008, 120.6932327747345]
    y_lbl = 'Execution time (sec)'
    title = 'Execution Time for Different fixed k values'

    linechart(x, y, x_lbl, y_lbl, title)

    #Silhouette Score
    y = [0.07499233, 0.086440551, 0.10323618, 0.12018893, 0.115965367]
    y_lbl = 'Silhouette Score'
    title = 'Silhouette Score for Different fixed k values'

    linechart(x, y, x_lbl, y_lbl, title)

    #Calinski-Harabasz Index:
    y = [104.281275, 119.3725409, 148.4243176, 167.7747748, 163.699444]
    y_lbl = 'Calinski-Harabasz Index'
    title = 'Calinski-Harabasz Index for Different fixed k values'

    #Davies-Bouldin Index:
    y = [3.939872274, 3.833204062, 3.358715647, 3.28947837, 3.293362661]
    y_lbl = 'Davies-Bouldin Index'
    title = 'Davies-Bouldin Index for Different fixed k values'

    #Average Clustering Coefficient:
    y = [0.257377312, 0.392679065, 0.431541357, 0.474004313, 0.517348139]
    y_lbl = 'Average Clustering Coefficient'
    title = 'Average Clustering Coefficient for Different fixed k values'

    #Average Cluster Density:
    y = [0.008876482, 0.020569198, 0.037580209, 0.06932726, 0.106114904]
    y_lbl = 'Average Clustering Density'
    title = 'Average Clustering Density for Different fixed k values'

    #Modularity
    y = [0.759738243, 0.718291132, 0.638713694, 0.613468871, 0.574966778]
    y_lbl = 'Modularity'
    title = 'Modularity for Different fixed k values'

    '''
    Dynamic k value
    '''

    x = [0.1, 0.3, 0.5, 0.75]
    x_lbl = 'k value as degree percentage'

    #Execution time:
    y = [1162.0318,1239.160426,1181.5826,1126.2623]
    y_lbl = 'Execution time (sec)'
    title = 'Execution Time for Different degree percentages'

    #Silhouette Score:
    y = [0.407848169,0.127570387,0.078779631,0.102067669]
    y_lbl = 'Silhouette Score'
    title = 'Silhouette Score for Different degree percentages'

    #Calinski-Harabasz Index:
    y = [66.56120197,153.6652488,137.5600241,152.8921214]
    y_lbl = 'Calinski-Harabasz Index'
    title = 'Calinski-Harabasz Index for Different degree percentages'

    #Davies-Bouldin Index:
    y = [3.677469867, 3.063563473, 2.954792409, 2.924797194]
    y_lbl = 'Davies-Bouldin Index'
    title = 'Davies-Bouldin Index for Different degree percentages'

    #Average Clustering Coefficient:
    y = [0.084300569,0.217787883,0.285645813,0.358458864]
    y_lbl = 'Average Clustering Coefficient'
    title = 'Average Clustering Coefficient for Different degree percentages'

    #Average Cluster Density:
    y = [0.08638607, 0.032525321, 0.04543224, 0.053720427]
    y_lbl = 'Average Clustering Density'
    title = 'Average Clustering Density for Different degree percentages'

    #Modularity:
    y = [0.426352124, 0.604726738, 0.626864398, 0.610880543]
    y_lbl = 'Modularity'
    title = 'Modularity for Different degree percentages'










if __name__ == "__main__":
    main()