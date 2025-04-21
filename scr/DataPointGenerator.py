#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 18.04.2025
# Version			: 1.0
#==============================================================================

# Import libaries
import pandas as pd
import numpy as np

class DataPointGenerator:
     """"
        # Class to generate data points for DBSCAN algorithm.

        This class visualized the result from the DBSCAN Clustering algorithm.

        ## Attribute(s):
        - none

        ## Method(s):
        - 'plot_sales()': returns a data frame of a timeline of sales per month
       

        ## Dependencies:
        - pandas
        - matplotlip
     """

     def __init__(self):
        """""
        ## Constuctor of DBSCANVisualizer class

        The constuctor takes on the root handler for the graphical user interface
        in order to display an error message box in case of an error in handling
        the csv file.

        # Parameter(s):
        - 'p_root' (tkinter root handle): handle for the graphical user interface

        # Return:
        - none
        """
    
     def generate_clusters_noise(self, num_clusters, num_points, num_noise_points):
        """
        ## This function will generate x, y coordinates organized in clusters with additional noise points.

            This function will serve as a data generator. 

        # Parameter(s):
        - num_clusters (int): Number of clusters in data set.
        - num_points (int): Number of data points per cluster.
        - num_noise_points (int): Number of noise points (randomly distributed).

        # Return:
        - DataFrame (pandas DataFrame): DataFrame containing numeric values of x, y coordinates.
        """
        clusters = []
        # Generate clusters
        for _ in range(num_clusters):
            # Generate random center for the cluster
            center_x, center_y = np.random.uniform(1, 50, 2)
            # Generate points around the center with some random noise
            points_x = np.random.normal(center_x, 1, num_points)
            points_y = np.random.normal(center_y, 1, num_points)
            clusters.append((points_x, points_y))

        # Generate noise points
        noise_x = np.random.uniform(1, 50, num_noise_points)
        noise_y = np.random.uniform(1, 50, num_noise_points)
        noise_points = (noise_x, noise_y)

        # Combine clusters and noise points into DataFrame
        df_cluster = {'x': [], 'y': []}
        for points_x, points_y in clusters:
            df_cluster['x'].extend(points_x.tolist())
            df_cluster['y'].extend(points_y.tolist())
    
        # Add noise points
        df_cluster['x'].extend(noise_points[0].tolist())
        df_cluster['y'].extend(noise_points[1].tolist())

        return pd.DataFrame(df_cluster)
        
        
     def generate_clusters(self, num_clusters, num_points):
        """"
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





