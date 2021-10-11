# Importing the libraries
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import plotly.graph_objs as graph_obj
from scipy.cluster import hierarchy

def display_elbow_method(x, clusters):
    """
    Displays an elbow graph for the given data x for the range of clusters specified
    :param x: array of data
    :param clusters: integer of max number of clusters to look at
    """
    inertia_scores = []
    cluster_range = range(1, clusters)

    # Gathering WCSS for range of clusters
    for i in cluster_range:
        model = KMeans(n_clusters=i)
        model.fit(x)
        inertia_scores.append(model.inertia_)

    # Plotting
    plt.plot(cluster_range, inertia_scores)
    plt.xlabel("Range of Clusters {}-{}".format(min(cluster_range), max(cluster_range)))
    plt.ylabel("WCSS Score")
    plt.title("Elbow Method")
    plt.show()

def display_dendogram(x):
    Z = hierarchy.linkage(x)
    hierarchy.dendrogram(Z)
    plt.show()

def display_3dplot(model, x, xaxis, yaxis, zaxis):
    """
    Plotting the clusters predicted by the model with the given data and labels for each axis
    :param model: cluster model
    :param x: array of data
    :param xaxis: string of x axis label
    :param yaxis: stirng of y axis label
    :param zaxis: string of z axis label
    """
    # Labels
    Scene = dict(xaxis=dict(title=xaxis), yaxis=dict(title=yaxis),
                 zaxis=dict(title=zaxis))

    # Clusters
    labels = model.labels_

    # Plot
    trace = [graph_obj.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',
                                marker=dict(color=labels, size=10, line=dict(color='black', width=10)))]
    layout = graph_obj.Layout(margin=dict(l=0, r=0), scene=Scene, height=800, width=800)
    fig = graph_obj.Figure(data=trace, layout=layout)
    fig.show()


if __name__ == "__main__":
    # Import the dataset
    dataset = pd.read_csv('Mall_Customers.csv')
    x = dataset.iloc[:, 2:].values

    # Finding optimal clusters
    # display_elbow_method(x, 10)
    # display_dendogram(x)

    optimal_no_of_clusters = 6

    # Training KMeans Model
    kmeans_model = KMeans(n_clusters=optimal_no_of_clusters)
    # kmeans_model.fit_predict(x)

    # Training Agglomerative Model
    agglomerative_cluster_model = AgglomerativeClustering(n_clusters=optimal_no_of_clusters)
    agglomerative_cluster_model.fit_predict(x)

    display_3dplot(agglomerative_cluster_model, x, 'Age -->', 'Spending Score -->', 'Annual Income -->')






