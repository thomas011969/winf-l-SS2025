
#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 20.04.2025
# Version			: 1.0
#==============================================================================
from DBSCANController import DBSCANController

#from flask import Flask, request, jsonify
#from flask_cors import CORS
import pandas as pd
from colorama import init, Fore, Back, Style

# Filename: bank-additional-full.csv
# Source Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
# Download link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# alternative download link: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data

jsonFilenameFlask = './csv/inputFlask.json'
jsonDataFilename = './csv/bank-additional-full-encoded.json'
jsonMappingFilename = './csv/bank-additional-full-encoding-map.json'

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
# parameters for DBSCAN
eps = 0.5
min_samples = 5
# parameters for data generation
num_clusters = 3
num_noise = 20
num_points = 100
# initialize colorama
init()

# create a json file from csv file
#df = pd.read_csv(filename, delimiter=";")
# convert into json
#json_file = './csv/bank-additional-full.json'
#df.to_json(json_file, orient='records', lines=True)
#print(Fore.GREEN + "JSON file created successfully!" + Style.RESET_ALL)

# create an instance from DBSCANController class
controller = DBSCANController(jsonFilenameFlask, jsonDataFilename, jsonMappingFilename)
# ---------------------------------------------------------------------------------
# - Flask REST API for DBSCAN
# ---------------------------------------------------------------------------------
# Initialize Flask app
#app = Flask(__name__)
#CORS(app)  # Aktiviert CORS f�r alle Routen
#@app.route('/dbscan', methods=['POST'])
#json_data_1 = controller.DBSCAN_from_file()
#print(json_data_1)
json_data_2 = controller.DBSCAN_scikit_from_file()
print(json_data_2)
#controller.run_comparison_animation(json_data_1, json_data_2)
controller.run_single_animation(json_data_2)
#json_data = controller.DBSCAN_data_generation(eps, min_samples, num_clusters, num_points, num_noise)
#controller.run_single_animation(json_data)
#print(json_data)