#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 13.04.2025
# Version			: 1.0
#==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class DBSCANAlgorithm:
    """"
        # Implementation of the DBSCAN Algorithm

        This class implements the DBSCAN algorithm.

        ## Attribute(s):
        - none

        ## Method(s):
        - 
       

        ## Dependencies:
        - pandas
        - time
        - numpy
        - sklearn
        - json
    """

    def __init__(self):
        pass

    def euklDistance(self, p1, p2):
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

    def region_query(self, data, point_idx, eps):
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
            if self.euklDistance(data[point_idx], data[i]) < eps:
                neighbors.append(i)
        return neighbors

    def convert_json_to_numpy(self, json_data):
        """
            ## Function to convert json data to numpy array
            This function takes a json data and converts it to a numpy array.
            # Parameter(s):
            - 'json_data' (json): json data to be converted
            # Return:
            - 'data' (numpy array): numpy array with all data points
        """
        # convert json data to pandas dataframe
        #df = pd.DataFrame(json_data)
        # convert pandas dataframe to numpy array
        #data = df.to_numpy()
        

    def dbscan(self, data, eps, min_samples):
        """
            ## This function implements the state machine for the DBSCAN algorithm

            This function runs the DBSCAN algorithm.

            # Parameter(s):
            - 'data' (numpy array): a numpy array with all data points to be evaluated
            - 'min_samples': minimum number of samples required to from a cluster
            - 'eps': maximum distance for neighbor points

            # Return:
            - 'cluster' (nump array): array with clusters
        """
        # create an numpy array by the name of label with the equal length of the data frame data
        # and fill the array with -1
        cluster = np.full(len(data), -1)
        print(eps)
        print(min_samples)
        print(len(data))

        # first cluster id equals zero
        cluster_id = 0
        # visit every data point
        for i in range(len(data)):
            # if data point is noise the move on to the next point
            if cluster[i] != -1:
                continue
            # if it data point is not noise than find neighbors
            neighbors = self.region_query(data, i, eps)
            # if the number of neighbors does not meet the number of minimum samples,
            # than mark this point as noise
            if len(neighbors) < min_samples:
                # mark data point as noise
                cluster[i] = -1  
            else:
            # assign an id to the cluster
                cluster[i] = cluster_id
                self.expand_cluster(data, cluster, i, neighbors, cluster_id, eps, min_samples)
                cluster_id += 1
        return cluster


    def expand_cluster(self, data, cluster, point_idx, neighbors, cluster_id, eps, min_samples):
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
                new_neighbors = self.region_query(data, neighbor_idx, eps)
                # if the new_neighbor meets the min_sample criteria, add the new
                # neighbor to the neighbor
                if len(new_neighbors) >= min_samples:
                    neighbors += new_neighbors
            i += 1
        print("done!")





