import matplotlib.pyplot as plt
import numpy as np

def linechart(x_values, y_values, x_label, y_label, title):

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', color='skyblue', linestyle='-')

    #labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x_values)
    #plt.ylim(100, 140)

    plt.show()

def barchart(x_values, y_values, x_label, y_label, title):
    #create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, color='skyblue')

    #labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(x_values)
    #plt.ylim(100, 140)

    plt.show()


def barchart_multi(x_values, y_values1, y_values2, y_values3, x_label, y_label, title):

    plt.figure(figsize=(10, 6))

    bar_width = 0.25

    #positions of the bars on the x-axis
    r1 = np.arange(len(x_values))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    #r4 = [x + bar_width for x in r3]

    #plot bars
    plt.bar(r1, y_values1, color='skyblue', width=bar_width, edgecolor='grey', label='Small G')
    plt.bar(r2, y_values2, color='green', width=bar_width, edgecolor='grey', label='Medium G')
    plt.bar(r3, y_values3, color='red', width=bar_width, edgecolor='grey', label='Large G')

    #Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks([r + bar_width for r in range(len(x_values))], x_values)

    plt.legend()

    plt.show()


def linechart_multi(x_values, y_values1, y_values2, y_values3, x_label, y_label, title):
    plt.figure(figsize=(10, 6))

    x_numeric = np.arange(len(x_values))

    #plots
    plt.plot(x_numeric, y_values1, marker='o', color='skyblue', linestyle='--', label='Small G')
    plt.plot(x_numeric, y_values2, marker='s', color='green', linestyle='--', label='Medium G')
    plt.plot(x_numeric, y_values3, marker='^', color='red', linestyle='--', label='Large G')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.xticks(x_numeric, x_values)

    plt.legend()

    plt.show()

def main():

    '''
    Below we have multiple calls of the functions above so that we may create plots for the interaction
    of all metrics with each k value, be it fixed or dynamic.
    '''

    x = ['2', '5', '10', '20', '30', 'Pure']
    x_lbl = 'k value'

    #Execution time
    set1 = [0.893477, 1.0960903, 1.316651, 1.28542, 1.5226564, 1.2]
    set2 = [131.1303973197937, 112.84199, 114.57211, 130.83008, 120.6932327747345, 4]
    set3 = [905, 1107, 1234, 1476, 1246, 16]


    y_lbl = 'Execution time (sec)'
    title = 'Execution Time for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Silhouette Score
    set1 = [0.116369394, 0.060162603, 0.053977545, 0.061019022, 0.061576692, 0.099709636]
    set2 = [0.095108505, 0.034738254,	-0.016912908, 0.008542022, -0.126682664, 0.054499063]
    set3 = [0.07499233, 0.086440551, 0.10323618, 0.12018893, 0.115965367, 0.100771844]

    y_lbl = 'Silhouette Score'
    title = 'Silhouette Score for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Calinski-Harabasz Index:
    set1 = [8.982709493, 5.96221004, 6.896940476, 7.633519393, 8.163762677, 9.592561953]
    set2 = [58.31603907, 38.20580347, 35.74507866, 41.9959374, 45.59880267, 33.90773006]
    set3 = [104.281275, 119.3725409, 148.4243176, 167.7747748, 163.699444, 147.5567181]

    y_lbl = 'Calinski-Harabasz Index'
    title = 'Calinski-Harabasz Index for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Davies-Bouldin Index:

    set1 = [2.49773806, 2.844542735, 2.892142938, 2.502110129, 2.514577407, 3.086808592]
    set2 = [3.324815232, 3.955922509, 3.898374179, 3.645531727, 3.694230662, 3.930861745]
    set3 = [3.939872274, 3.833204062, 3.358715647, 3.28947837, 3.293362661, 3.38610154]

    y_lbl = 'Davies-Bouldin Index'
    title = 'Davies-Bouldin Index for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Average Clustering Coefficient:
    set1 = [0.089630018,			0.357024141,		0.410800753,	0.386453333,	0.463207216, 0.535667709]
    set2 = [0.258938827,			0.39236328,			0.386443328,		0.396368131,		0.399235929, 0.144785715]
    set3 = [0.257377312, 0.392679065, 0.431541357, 0.474004313, 0.517348139, 0.400015358]

    y_lbl = 'Average Clustering Coefficient'
    title = 'Average Clustering Coefficient for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Average Cluster Density:
    set1 = [0.161549178,			0.305955492,			0.308413063,		0.376465756,		0.442097316, 0.438621119]
    set2 = [0.022401868,			0.029274357,			0.051394588,		0.076232874,		0.084666358, 0.036844297]
    set3 = [0.008876482, 0.020569198, 0.037580209, 0.06932726, 0.106114904, 0.043585089]

    y_lbl = 'Average Clustering Density'
    title = 'Average Clustering Density for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Modularity
    set1 = [0.393957117,			0.297279168,			0.285769476,		0.216404538,		0.185417077, 0.27871803]
    set2 = [0.573664522,			0.303502041,			0.281942583,		0.238383802,		0.157223181, 0.216]
    set3 = [0.759738243, 0.718291132, 0.638713694, 0.613468871, 0.574966778, 0.647644972]

    y_lbl = 'Modularity'
    title = 'Modularity for Different fixed k values'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    '''
    Dynamic k value
    '''

    x = ['0.1', '0.3', '0.5', '0.75', 'Pure']
    x_lbl = 'k value as degree percentage'

    #Execution time:
    set1 = [0.980987, 1.0907225608825684,			1.244810,		1.32748818397,		1.217379]
    set2 = [144.96753025054932,			116.45659470558167,			115.99030,		4.66582,		4.436213493347168]
    set3 = [1162.0318, 1239.160426, 1181.5826, 1126.2623, 16.30583]

    y_lbl = 'Execution time (sec)'
    title = 'Execution Time for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Silhouette Score:
    set1 = [0.162441092,			0.060574317,			-0.041093002,		0.055112602,		0.099709636]
    set2 = [0.404000892,			0.133501096,			-0.00088554,		-0.046185468,		0.054499063]
    set3 = [0.407848169,0.127570387,0.078779631,0.102067669,0.100771844]

    y_lbl = 'Silhouette Score'
    title = 'Silhouette Score for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Calinski-Harabasz Index:
    set1 = [7.914353925,			6.478652022,			7.915030634,		9.117622737,		9.592561953]
    set2 = [13.58832808	,		10.22876185		,	10.70793112	,	27.6471539,		33.90773006]
    set3 = [66.56120197,153.6652488,137.5600241,152.8921214,147.5567181]

    y_lbl = 'Calinski-Harabasz Index'
    title = 'Calinski-Harabasz Index for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Davies-Bouldin Index:
    set1 = [2.711990531,			2.932096977	,		2.8070057	,	2.775119553,		3.086808592]
    set2 = [3.396648384,			3.244573595,			2.822624381,		3.787347478,		3.930861745]
    set3 = [3.677469867, 3.063563473, 2.954792409, 2.924797194, 3.38610154]
    y_lbl = 'Davies-Bouldin Index'
    title = 'Davies-Bouldin Index for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Average Clustering Coefficient:
    set1 = [0.041347551	,		0.318965505	,		0.356804681	,	0.548646169	,	0.535667709]
    set2 = [0.009954676	,		0.017426357	,		0.031037399	,	0.088395299,		0.144785715]
    set3 = [0.084300569,0.217787883,0.285645813,0.358458864,0.400015358]
    y_lbl = 'Average Clustering Coefficient'
    title = 'Average Clustering Coefficient for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Average Cluster Density:
    set1 = [0.230066666,			0.313104401,			0.379140043,		0.464933762,		0.438621119]
    set2 = [0.073405043,			0.130164842	,		0.17939985	,	0.065824791	,	0.036844297]
    set3 = [0.08638607, 0.032525321, 0.04543224, 0.053720427, 0.043585089]
    y_lbl = 'Average Clustering Density'
    title = 'Average Clustering Density for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

    #Modularity:
    set1 = [0.286386713,			0.222503213,			0.274727911,		0.241423602,		0.27871803]
    set2 = [0.105377287,			0.128977573,			0.105613425,		0.205631405,		0.216990343]
    set3 = [0.426352124, 0.604726738, 0.626864398, 0.610880543, 0.647644972]
    y_lbl = 'Modularity'
    title = 'Modularity for Different degree percentages'

    linechart_multi(x, set1, set2, set3, x_lbl, y_lbl, title)

if __name__ == "__main__":
    main()
