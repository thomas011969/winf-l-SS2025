#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 13.04.2025
# Version			: 3.0
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
import json


def euklDistance(p1, p2):
    """
        ## A function to calculate the ekledian distance between two points

        This function takes a data frame an extracs the sales per month data

        # Parameter(s):
        - 'p1' (numieric value): point 1
        - 'p2' (numeric value): point 2

        # Return:
        - 'distance' (numeric value): euklidian distance between p1 and p2
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))

def region_query(data, point_idx, eps):
    """
        ## A function to find neighbors of point_idx.

        This function takes a data frame with data points and evaluates which points in this data frame
        are within the distance eps of the given data point.

        # Parameter(s):
        - 'data' (data frame): data frame with all data points to be evaluated
        - 'point_idx': data point for which neighbors have to be found
        - 'eps': maximum distance for neighbor points

        # Return:
        - 'neighbors': a list with neighbors from point_idx which are within the distancse of eps
    """
    neighbors = []
    for i in range(len(data)):
        if euklDistance(data[point_idx], data[i]) < eps:
            neighbors.append(i)
    return neighbors


def dbscan(data, eps, min_samples):
    """
        ## This function implements the state machine for the DBSCAN algorithm

        This function runs the DBSCAN algorithm.

        # Parameter(s):
        - 'data' (data frame): data frame with all data points to be evaluated
        - 'min_samples': minimum number of samples required to from a cluster
        - 'eps': maximum distance for neighbor points

        # Return:
        - 'cluster' (nump array): array with clusters
    """
    # create an numpy array by the name of label with the equal length of the data frame data
    # and fill the array with -1
    cluster = np.full(len(data), -1)
    # first cluster id equals zero
    cluster_id = 0
    # visit every data point
    for i in range(len(data)):
        # if data point is noise the move on to the next point
        if cluster[i] != -1:
            continue
        # if it data point is not noise than find neighbors
        neighbors = region_query(data, i, eps)
        # if the number of neighbors does not meet the number of minimum samples,
        # than mark this point as noise
        if len(neighbors) < min_samples:
            # mark data point as noise
            cluster[i] = -1  
        else:
            # assign an id to the cluster
            cluster[i] = cluster_id
            expand_cluster(data, cluster, i, neighbors, cluster_id, eps, min_samples)
            cluster_id += 1
    return cluster


def expand_cluster(data, cluster, point_idx, neighbors, cluster_id, eps, min_samples):
    """
        ## This function expands the cluster

        This function expandes the cluster, by identifying all data points which can
        be assigned to the cluster.

        # Parameter(s):
        - 'data' (data frame): data frame with all data points to be evaluated
        - 'eps': maximum distance for neighbor points
        - 'neighbors': a list with neighbors from point_idx which are within the distancse of eps
        - 'point_idx': data point for which neighbors have to be found
        - 'cluster_id': ID of current cluster
        - 'min_samples': minimum number of samples required to from a cluster

        # Return:
        - none
    """
    i = 0
    # iterate through all neighbors
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        # if a neighbor is marked a noise than assign it to clsuter
        if cluster[neighbor_idx] == -1:
            cluster[neighbor_idx] = cluster_id
            # find the neighbor of the neighbor
            new_neighbors = region_query(data, neighbor_idx, eps)
            # if the new_neighbor meets the min_sample criteria, add the new
            # neighbor to the neighbor
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors
        i += 1


def show_animation(xy, cluster, sci_cluster, num_clusters, elapsed_time, sci_elapsed_time, sci_num_clusters):
    """
    ## This function will animate the result out of the DBSCAN

    This function will serve as an animation.

    # Parameter(s):
    - xy: Data points as a dictionary with 'x' and 'y' keys
    - cluster: Cluster cluster for the data points

    # Return:
    - none
    """
    # create a windwow for two plots
    fig, (ax4, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 6))

    # put text string into a box
    textstr = f'Elapsed Time custom: {elapsed_time:.4f} seconds\nNumber of Clusters custom: {num_clusters}\nElapsed Time sci: {sci_elapsed_time:.4f} seconds\n Number of Clusters sci: {sci_num_clusters}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.5, 0.5, textstr, transform=ax4.transAxes, fontsize=14,
                verticalalignment='center', horizontalalignment='center', bbox=props)
    # Achsen ausblenden
    ax4.axis('off')


    # create plot for original data
    ax1.plot(xy['x'], xy['y'], 'k+', markersize=10)
    ax1.set_title('Original Data')
    #ax1.legend([textstr], loc='upper left', fontsize=12, frameon=True, fancybox=True, framealpha=0.5)

    # create second plot with data from DBSCAN custom algorithm
    unique_cluster = set(cluster)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_cluster))]
    # iterate through clusters and form a tuple with (clusterid, color)
    # col is an array containing the color, coded in RGB [R, G, B, Opacity]
    for k, col in zip(unique_cluster, colors):
        # handle noise data points
        if k == -1:
            continue    # hide noise points. If they should be display, comment out this line!
            # create a RGB array for black [R, G, B, Opacity]
            black = [0, 0, 0, 1]
            # create a mask for all data points which are noise
            # True would mean, data point is noise
            class_member_mask = (cluster == k)
            # array with noise data. This array contains only the data points which where masked 
            # as noise.
            xy_class = np.array([xy['x'][class_member_mask], xy['y'][class_member_mask]]).T
            ax2.plot(xy_class[:, 0], xy_class[:, 1], '*', markerfacecolor=tuple(black), markersize=10)
        else:
            class_member_mask = (cluster == k)
            xy_class = np.array([xy['x'][class_member_mask], xy['y'][class_member_mask]]).T
            ax2.plot(xy_class[:, 0], xy_class[:, 1], 'o', markerfacecolor=tuple(col), markersize=6)
        ax2.set_title('DBSCAN Clusters Custom')
        plt.pause(.5)

    # create third plot with data from DBSCAN sci-kit algorithm
    unique_cluster_sci = set(sci_cluster)
    #colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_cluster_sci))]
    # iterate through clusters and form a tuple with (clusterid, color)
    # col is an array containing the color, coded in RGB [R, G, B, Opacity]
    for k, col in zip(unique_cluster_sci, colors):
        # handle noise data points
        if k == -1:
            continue    # hide noise points. If they should be display, comment out this line!
            # create a RGB array for black [R, G, B, Opacity]
            black = [0, 0, 0, 1]
            # create a mask for all data points which are noise
            # True would mean, data point is noise
            class_member_mask = (cluster == k)
            # array with noise data. This array contains only the data points which where masked 
            # as noise.
            xy_class = np.array([xy['x'][class_member_mask], xy['y'][class_member_mask]]).T
            ax3.plot(xy_class[:, 0], xy_class[:, 1], '*', markerfacecolor=tuple(black), markersize=10)
        else:
            class_member_mask = (sci_cluster == k)
            xy_class = np.array([xy['x'][class_member_mask], xy['y'][class_member_mask]]).T
            ax3.plot(xy_class[:, 0], xy_class[:, 1], 'o', markerfacecolor=tuple(col), markersize=6)
        ax3.set_title('DBSCAN Clusters sci-kit')
        plt.pause(.5)
    plt.show()

def generate_clusters(num_clusters, num_points):
    """
        ## This function will generate x,y coordinates organized in clusters

        This function will serve as a data generator. 

        # Parameter(s):
        - 'num_clusters' (int): Number of clusters in data set
        - 'num_points' (int): Number of data points.

        # Return:
        - data frame (pandas data frame): data frame containing numeric values of xy coordinates
    """
    clusters = []
    # generate numpy array with coordinates
    for _ in range(num_clusters):
        # Generate random center for the cluster
        center_x, center_y = np.random.uniform(1, 50, 2)
        # Generate points around the center with some random noise
        points_x = np.random.normal(center_x, 1, num_points)
        points_y = np.random.normal(center_y, 1, num_points)
        # Add some outliers
        outliers_x = np.random.uniform(1, 50, int(num_points * 0.05))
        outliers_y = np.random.uniform(1, 50, int(num_points * 0.05))
        points_x = np.concatenate([points_x, outliers_x])
        points_y = np.concatenate([points_y, outliers_y])
        clusters.append((points_x, points_y))
    # converted numpy array to pandas data frame
    df_cluster = {'x': [], 'y': []}
    for points_x, points_y in clusters:
        df_cluster['x'].extend(points_x.tolist())
        df_cluster['y'].extend(points_y.tolist())
    return pd.DataFrame(df_cluster) 


# ---------------------------------------------------------------------------------
# - Set parameters for DBSCAN and data generation
# ---------------------------------------------------------------------------------
eps = 0.5
min_samples = 5
num_clusters = 5
num_points = 20

## ---------------------------------------------------------------------------------
# - generate data to run DBSCAN
# ---------------------------------------------------------------------------------
xy = generate_clusters(num_clusters, num_points)
# Convert DataFrame to numpy array
data = xy[['x', 'y']].values

# ---------------------------------------------------------------------------------
# - run DBSCAN algorithm custom implementation
# ---------------------------------------------------------------------------------
# Run DBSCAN
start_time = time.time()
cluster = dbscan(data, eps, min_samples)
end_time = time.time()
# calculate elapsed time to run DBSCAN
elapsed_time = end_time - start_time
# calculate the number of cluster without noise
# substract one in case value in cluster is a noise point, other
# substract 0
num_clusters = len(set(cluster)) - (1 if -1 in cluster else 0)
print(f"Number of clusters custom algorithm = {num_clusters}")
print(f"Time elapsed for DBSCAN custom algorithm = {elapsed_time:.4f} seconds!")

# ---------------------------------------------------------------------------------
# - run DBSCAN algorithm from SciLearn library
# ---------------------------------------------------------------------------------
# get start time for running scikit-learn DBSCAN
start_time = time.time()
# run DBSCAN from scikit-learn
db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
# stopp time measurement
end_time = time.time()
# calculate elapsed time for DBSCAN sci-kit
sci_elapsed_time = end_time - start_time

# extract clusters
sci_cluster = db.labels_
print(type(sci_cluster))
# calculate the number of cluster without noise
# substract one in case value in cluster is a noise point, other
# substract 0
sci_num_clusters = len(set(cluster)) - (1 if -1 in cluster else 0)
print(f"Number of clusters scikit-learn algorithm = {sci_num_clusters}")
print(f"Time elapsed for DBSCAN scikit-learn algorithm = {sci_elapsed_time:.4f} seconds!")


# ---------------------------------------------------------------------------------
# - run animation
# ---------------------------------------------------------------------------------
#show_animation(xy, cluster, sci_cluster, num_clusters, elapsed_time, sci_elapsed_time, sci_num_clusters)


# ---------------------------------------------------------------------------------
# - create a json file
# ---------------------------------------------------------------------------------
cluster_custom = cluster.tolist()
j_cluster_str = json.dumps(cluster_custom)
#print(j_cluster_str)
xy['cluster'] = cluster.tolist()
df_str = xy.to_json(orient='records', lines=True)
with open('xy.json', 'w') as json_file:
    json_file.write(df_str)
