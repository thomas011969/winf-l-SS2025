#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 18.04.2025
# Version			: 1.0
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
import json
from DBSCANAlgorithm import DBSCAN
from  DataPointGenerator import DataPointGenerator
from DBSCANVisualizer import DBSCANVisualizer

class DBSCANController:
    # ---------------------------------------------------------------------------------
    # - Set parameters for DBSCAN and data generation
    # ---------------------------------------------------------------------------------
    # Data for DBSCAN algorithm
    eps = 0.5
    min_samples = 5
    # Parameters for data generation
    num_clusters = 5
    num_points = 20

    def __init__(self):
        """
        ## Constuctor of the DBSCANController class

        Constructor of class. 

        # Parameter(s):
        - none

        # Return:
        - none
        """
        self.dbscan = DBSCAN()

    def run_DBSCAN(self, p_np_cluster_points, p_eps, p_min_samples):
        """"
            ## This function will run the DBSCAN algorithm 

            This function will run the DBSCAN algorithm and measure the time
            it takes to process the data

            # Parameter(s):
            - 'p_np_cluster_points' (numpy array): a numpy array containing the xy-coordinates
            - 'p_min_samples': minimum number of samples required to from a cluster
            - 'p_eps': maximum distance for neighbor points

            # Return:
            - 'np_cluster' : a numpy array containing the cluster id for each xy-ccordinate
        """
        # ---------------------------------------------------------------------------------
        # - run DBSCAN algorithm custom implementation
        # ---------------------------------------------------------------------------------
        # Run DBSCAN
        start_time = time.time()
        np_cluster = self.dbscan(p_np_cluster_points, p_eps, p_min_samples)
        end_time = time.time()
        # calculate elapsed time to run DBSCAN
        elapsed_time = end_time - start_time
        # calculate the number of cluster without noise
        # substract one in case value in cluster is a noise point, other
        # substract 0
        num_clusters = len(set(np_cluster)) - (1 if -1 in np_cluster else 0)
        print(f"Number of clusters custom algorithm = {num_clusters}")
        print(f"Time elapsed for DBSCAN custom algorithm = {elapsed_time:.4f} seconds!")
        return np_cluster
        
    def DBSCAN_from_file(self, p_filename, p_eps, p_min_samples):
        """"
            ## This function will run the DBSCAN algorithm using self generated data points

            This function will first generate data points usinge the class DataPointGenerator,
            and than run the DBSCAN algorithm based on the data points

            # Parameter(s):
            - 'p_filename' (str): the filename for the csv file with xy-coordinates
            - 'p_min_samples': minimum number of samples required to from a cluster
            - 'p_eps': maximum distance for neighbor points

            # Return:
            - json object : data frame containing numeric values of xy coordinates
                            and the assigned cluster id for each data point.
        """
        ## ---------------------------------------------------------------------------------
        # - generate data to run DBSCAN
        # ----------------------------------------------------------------------------------
        print("This is DBSCAN using using a .csv file with xy-data points!")
        # read csv file and store it in a data frame
        df_cluster_points = pd.read_csv(p_filename)
        # Convert DataFrame to numpy array
        np_cluster_points = df_cluster_points[['x', 'y']].values
        # run DBSCAN
        np_cluster = self.run_DBSCAN(self, np_cluster_points, p_eps, p_min_samples)        
        # create json object
        # add cluster id to data frame with xy coordinates
        df_cluster_points['cluster'] = np_cluster.tolist()
        # create a json object
        json_str = df_cluster_points.to_json(orient='records', lines=True)
        return json_str

    def DBSCAN_data_generation(self, p_eps, p_min_samples, p_num_clusters, p_num_points):
        """"
            ## This function will run the DBSCAN algorithm using self generated data points

            This function will first generate data points usinge the class DataPointGenerator,
            and than run the DBSCAN algorithm based on the data points

            # Parameter(s):
            - 'num_clusters' (int): Number of clusters in data set
            - 'num_points' (int): Number of data points.
            - 'p_min_samples' (int): minimum number of samples required to from a cluster
            - 'p_eps' (double): maximum distance for neighbor points

            # Return:
            - json object : data frame containing numeric values of xy coordinates
                            and the assigned cluster id for each data point.
        """
        ## ---------------------------------------------------------------------------------
        # - generate data to run DBSCAN
        # ----------------------------------------------------------------------------------
        print("This is DBSCAN using self generated xy-data points!")
        data_generator = DataPointGenerator()
        df_cluster_points = data_generator.generate_clusters(p_num_clusters, p_num_points)
        # Convert DataFrame to numpy array
        np_cluster_points = df_cluster_points[['x', 'y']].values
        # run DBSCAN
        np_cluster = self.run_DBSCAN(self, np_cluster_points, p_eps, p_min_samples)        
        # create json object
        # add cluster id to data frame with xy coordinates
        df_cluster_points['cluster'] = np_cluster.tolist()
        # create a json object
        json_str = df_cluster_points.to_json(orient='records', lines=True)
        return json_str


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
    #with open('xy.json', 'w') as json_file:
    #    json_file.write(df_str)
