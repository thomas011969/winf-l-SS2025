
#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 20.04.2025
# Version			: 1.0
#==============================================================================
from DBSCANController import DBSCANController

# standard filename for this application
# Data source loaded from: https://archive.ics.uci.edu/dataset/352/online+retail
# Citation: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
# The original .xlsx file was for the purpose of this programm converted to a .csv file using Excel.
filename = './csv/data_points.csv'

""""
    # Main function

    The purpose of the main function is to create an instance of the controller class
    and start the controller. The application is build according to the MVC 
    (Master-Viewer-Controller) design pattern. The controller class will create an
    instance of the data model and viewer class, and handle all events initiated form
    the graphical user insterface.

    ## Attribute(s):
    - none

    ## Method(s):
    - none
    
    ## Dependencies:
    - class controller
"""
eps = 0.5
min_samples = 5
num_clusters = 3
num_noise = 20
num_points = 100

controller = DBSCANController()
json_data_1 = controller.DBSCAN_from_file(filename, eps, min_samples)
json_data_2 = controller.DBSCAN_scikit_from_file(filename, eps, min_samples)
controller.run_comparison_animation(json_data_1, json_data_2)
#controller.run_single_animation(json_data)
#json_data = controller.DBSCAN_data_generation(eps, min_samples, num_clusters, num_points, num_noise)
#controller.run_single_animation(json_data)
#print(json_data)