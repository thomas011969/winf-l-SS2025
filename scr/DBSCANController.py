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
from DBSCANAlgorithm import DBSCANAlgorithm
from DataPointGenerator import DataPointGenerator
from DBSCANVisualizer import DBSCANVisualizer
from DBSCANJsonReader import DBSCANJsonReader

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

    def __init__(self, p_filename):
        """
        ## Constuctor of the DBSCANController class

        Constructor of class. 

        # Parameter(s):
        - none

        # Return:
        - none
        """
        self.dbscan = DBSCANAlgorithm()
        self.visualizer = DBSCANVisualizer()
        self.jreader = DBSCANJsonReader(p_filename)

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
        np_cluster = self.dbscan.dbscan(p_np_cluster_points, p_eps, p_min_samples)
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
        
    def DBSCAN_from_file(self):
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
        #df_cluster_points = pd.read_csv(p_filename)
        df_cluster_points = self.jreader.getData()
        eps = self.jreader.getEPS()
        min_samples = self.jreader.getMinSamples()
        headers = self.jreader.getHeaders()
        # Convert DataFrame to numpy array
        #np_cluster_points = df_cluster_points[['x', 'y']].values
        np_cluster_points = df_cluster_points[[headers[0], headers[1]]].values
        # run DBSCAN
        np_cluster = self.run_DBSCAN(np_cluster_points, eps, min_samples)        
        # create json object
        # add cluster id to data frame with xy coordinates
        df_cluster_points['cluster'] = np_cluster.tolist()
        # create a json object
        json_str = df_cluster_points.to_json(orient='records', lines=True)
        return json_str

    def DBSCAN_data_generation(self, p_eps, p_min_samples, p_num_clusters, p_num_points, p_num_noise):
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
        df_cluster_points = data_generator.generate_clusters_noise(p_num_clusters, p_num_points, p_num_noise)
        # Convert DataFrame to numpy array
        np_cluster_points = df_cluster_points[['x', 'y']].values
        # run DBSCAN
        np_cluster = self.run_DBSCAN(np_cluster_points, p_eps, p_min_samples)        
        # create json object
        # add cluster id to data frame with xy coordinates
        df_cluster_points['cluster'] = np_cluster.tolist()
        # create a json object
        json_str = df_cluster_points.to_json(orient='records', lines=True)
        return json_str

    def DBSCAN_scikit_from_file(self):
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
        # ---------------------------------------------------------------------------------
        # - run DBSCAN algorithm from SciLearn library
        # ---------------------------------------------------------------------------------
        print("This is DBSCAN SciKit using using a .csv file with xy-data points!")
        # read csv file and store it in a data frame
        #df_cluster_points = pd.read_csv(p_filename)
        # Convert DataFrame to numpy array

        df_cluster_points = self.jreader.getData()
        p_eps = self.jreader.getEPS()
        p_min_samples = self.jreader.getMinSamples()
        headers = self.jreader.getHeaders()
        # Convert DataFrame to numpy array
        #np_cluster_points = df_cluster_points[['x', 'y']].values
        np_cluster_points = df_cluster_points[[headers[0], headers[1]]].values


        #np_cluster_points = df_cluster_points[['x', 'y']].values
        start_time = time.time()
        # run DBSCAN from scikit-learn
        db = DBSCAN(eps=p_eps, min_samples=p_min_samples).fit(np_cluster_points)
        # stopp time measurement
        end_time = time.time()
        # calculate elapsed time for DBSCAN sci-kit
        sci_elapsed_time = end_time - start_time

        # extract clusters
        sci_cluster = db.labels_
        # calculate the number of cluster without noise
        # substract one in case value in cluster is a noise point, other
        # substract 0
        sci_num_clusters = len(set(sci_cluster)) - (1 if -1 in sci_cluster else 0)
        print(f"Number of clusters scikit-learn algorithm = {sci_num_clusters}")
        print(f"Time elapsed for DBSCAN scikit-learn algorithm = {sci_elapsed_time:.4f} seconds!")
        df_cluster_points['cluster'] = sci_cluster.tolist()
        # create a json object
        json_str = df_cluster_points.to_json(orient='records', lines=True)
        return json_str

    def run_single_animation(self, p_json_data):
        """"
            ## This function will run an animation with a single data set

            This function will dispolay the result of the dbscan algorithm

            # Parameter(s):
            - 'p_json_data' (json object): a json object containing the xy and cluster id

            # Return:
            - none
        """
        # convert json data to a dictonary
        # extract line by line
        json_lines = p_json_data.splitlines()  
        json_object = [json.loads(line) for line in json_lines]

        # NumPy-Array for x and y
        keys = list(json_object[0].keys())
        x_key, y_key = keys[0], keys[1]
       
        # Create the NumPy array using the dynamic keys
        data_array = np.array([[item[x_key], item[y_key]] for item in json_object])
        #data_array = np.array([[item["x"], item["y"]] for item in json_object])

        # NumPy-Array for cluster
        cluster_array = np.array([item["cluster"] for item in json_object])
        
        self.visualizer.show_single_animation(data_array, cluster_array, keys)

    def run_comparison_animation(self, p_json_data_1, p_json_data_2):
        """"
            ## This function will run an animation with two data sets

            This function will dispolay the result of the dbscan algorithm with two data sets

            # Parameter(s):
            - 'p_json_data_1' (json object): a json object containing the xy and cluster id Data set 1
            - 'p_json_data_2' (json object): a json object containing the xy and cluster id Data set 2

            # Return:
            - none
        """
        # convert json data to a dictonary
        # extract line by line
        json_lines1 = p_json_data_1.splitlines()  
        json_object1 = [json.loads(line) for line in json_lines1]

        # NumPy-Array for x and y for data set 1
        # NumPy-Array for x and y
        keys1 = list(json_object1[0].keys())
        x_key1, y_key1 = keys1[0], keys1[1]
        # Create the NumPy array using the dynamic keys
        data_array_1 = np.array([[item[x_key1], item[y_key1]] for item in json_object1])
        #data_array_1 = np.array([[item["x"], item["y"]] for item in json_object])
        # NumPy-Array for cluster for data set 1
        cluster_array_1 = np.array([item["cluster"] for item in json_object1])

        # NumPy-Array for x and y for data set 2
        # convert json data to a dictonary
        # extract line by line
        json_lines2 = p_json_data_1.splitlines()  
        json_object2 = [json.loads(line) for line in json_lines2]

        # NumPy-Array for x and y for data set 1
        # NumPy-Array for x and y
        keys2 = list(json_object1[0].keys())
        x_key2, y_key2 = keys2[0], keys2[1]
        # Create the NumPy array using the dynamic keys
        data_array_2 = np.array([[item[x_key2], item[y_key2]] for item in json_object2])
        #data_array_2 = np.array([[item["x"], item["y"]] for item in json_object])
        # NumPy-Array for cluster for data set 1
        cluster_array_2 = np.array([item["cluster"] for item in json_object2])
        
        self.visualizer.show_comparison(data_array_1, cluster_array_1, data_array_2, cluster_array_2, keys1, keys2)