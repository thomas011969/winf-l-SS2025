#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 04.05.2025
# Version			: 1.0
#==============================================================================

# Import libaries
from re import I
import pandas as pd
import json

class DBSCANJsonReader:
    """"
        # Class to read json files.
        This class reads seperates data from parameters of a given json file
        ## Attribute(s):
        - none
        ## Method(s):
        - 'read_json()': returns a data frame of a timeline of sales per month
       
        ## Dependencies:
        - pandas

    """
    def __init__(self, p_filenameFlask, p_filenameData, p_filenameMapping):
        """""
        ## Constuctor of DBSCANJsonReader class
        The constuctor takes as a parameter a filename and reads the data and parameters
        # Parameter(s):
        - 'p_filenameData' (str): filename of data file in json format
        - 'p_filename' (str): filename of json file handed over by flask
        # Return:
        - none
        """
        # open data file an store data in a dataframe
        self.data = pd.read_json(p_filenameData)
        # open mapping file and store information in a dataframe
        with open(p_filenameMapping, "r") as file:
            self.mapping = json.load(file)
        # open flask file and extract data
        with open(p_filenameFlask, "r") as file:
            data = json.load(file)
        # store information from flask file in lists
        self.parameters = list(data.get("parameters", {}).values())
        self.categories = list(data.get("categories", {}).values())
        self.categorical_columns = [
            "job", "marital", "education", "default",
            "housing", "loan", "contact", "month", "day_of_week"
        ]
        self.categorical_enc_columns = [
            "job_encoded", "marital_encoded", "education_encoded", "default_encoded",
            "housing_encoded", "loan_encoded", "contact_encoded", "month_encoded", "day_of_week_encoded"
        ]

    def getEPS(self):
        """" 
            ## This function will return the eps parameter from the json file
             
            This function returns the eps parameter of the json file
             
            # Return:
            - eps (float) : eps parameter from the json file
        """
        return self.parameters[1]

    def getMinSamples(self):
        """" 
            ## This function will return the eps parameter from the json file
             
            This function returns the eps parameter of the json file
             
            # Return:
            - min_samples (int) : minimum samples per cluster
        """
        return self.parameters[2]
    
    def decodeCategories(self, p_jsonStr):
        """" 
            ## This function will return the decoded json string
             
            This function will return the decoded json string
             
            # Return:
            - json string : decoded json string
        """

        # revers mapping
        inverse_mapping = {
            category + "_encoded": {v: k for k, v in values.items()}
            for category, values in self.mapping.items()
        }
        # decode line by line
        rows = [json.loads(line) for line in p_jsonStr.splitlines() if line.strip()]
        
        # array for decoded data
        decoded_data = []

        for row in rows:
            decoded_row = {}
            for key, value in row.items():
                if key in inverse_mapping:  # Falls Schlüssel im Mapping existiert
                    decoded_row[key.replace("_encoded", "")] = inverse_mapping[key].get(value, value)
                else:
                    decoded_row[key] = value  # Falls kein Mapping existiert, unverändert lassen
            decoded_data.append(decoded_row)
        # save decoded data
        #with open("daten_decoded.json", "w") as file:
        #    json.dump(decoded_data, file, indent=4)
        return decoded_data

    def getData(self, p_categories):
        """" 
            ## This function will return the date from the bank data json file
             
            This function returns the data section of the bank data json file

            # Parameter(s):
            - p_categories (list) : a list of names of headers
             
            # Return:
            - df (pandas data frame) : data frame containing data to be analyzed
        """
        # Select only the columns specified in p_categories
        encoded_columns = [col + "_encoded" if col in self.categorical_columns else col for col in p_categories]
        data = self.data[encoded_columns]  
        return data

    def getListCategories(self):
        """" 
            ## This function will return the categories  from the json file as a list
             
            This function returns the categorie information from categories section of the json file
             
            # Return:
            - headers (list) : a list of names of headers
        """
        return self.categories


    def readParameters(self, p_filename):
        """"
            ## This function will read in the parameter section from the json file

            This function takes a filename as a parameter and returns the parameter section of
            the file

            # Parameter(s):
            - 'p_filename' (str): the filename for the csv file with xy-coordinates

            # Return:
            - parameter_dict (dictionary) : a dictionary containing the parameters  
        """
        with open(p_filename, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        parameters_dict = json_data.get("parameters", {})
        return parameters_dict

 
   



