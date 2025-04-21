
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
num_clusters = 5
num_points = 20

controller = DBSCANController()
json_file = controller.run_DBSCAN_from_file(filename, eps, min_samples)
print(json_file)