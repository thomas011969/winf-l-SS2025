#==============================================================================
# Title				: Wirtschaftsinformatik Labor
# Author 			: Thomas Schmidt
# Contact 			: thomas.schmidt.2@stud.hs-emden-leer.de
# Date				: 18.04.2025
# Version			: 1.0
#==============================================================================

# Import libaries
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
    def __init__(self, p_filename):
        """""
        ## Constuctor of DBSCANVisualizer class
        The constuctor takes on the root handler for the graphical user interface
        in order to display an error message box in case of an error in handling
        the csv file.
        # Parameter(s):
        - 'p_filename' (str): filename
        # Return:
        - none
        """
        self.data = self.readData(p_filename)
        self.parameters = self.readParameters(p_filename)

    def convertCSVtoJson(self, filename):
        # read csv file
        df = pd.read_csv(filename)
        # convert data frame to json
        json_data = df.to_json(orient="records")
        parameters = {
            "algorithm": "DBSCAN",
            "eps": 0.5,
            "min_samples": 5 
            }
        combined_data = {
            "parameters": parameters,
            "data":json_data 
            }
        print(combined_data)
        # save json file
        with open("data_1.json", "w", encoding="utf-8") as json_file:
            json.dump(combined_data, json_file)

        return combined_data

    def getEPS(self):
        """" 
            ## This function will return the eps parameter from the json file
             
            This function returns the eps parameter of the json file
             
            # Return:
            - eps (float) : eps parameter from the json file
        """
        return self.parameters["eps"]

    def getMinSamples(self):
        """" 
            ## This function will return the eps parameter from the json file
             
            This function returns the eps parameter of the json file
             
            # Return:
            - min_samples (int) : minimum samples per cluster
        """
        return self.parameters["min_samples"]

    def getData(self):
        """" 
            ## This function will return the data section from the json file
             
            This function returns the data section of the json file
             
            # Return:
            - df (pandas data frame) : data frame containing the data to be analyzed
        """
        return self.data

    def getHeaders(self):
        """" 
            ## This function will return the headers of the data section from the json file
             
            This function returns the header information from data section of the json file
             
            # Return:
            - headers (list) : a list of names of headers
        """
        headers = self.data.columns.tolist()  # Convert to a list
        return headers


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

    def readData(self, p_filename):
        """"
            ## This function will read in the data section from the json file

            This function takes a filename as a parameter and returns the data section of
            the file

            # Parameter(s):
            - 'p_filename' (str): the filename for the csv file with xy-coordinates

            # Return:
            - df (pandas data frame) : data frame containing the data to be analyzed
        """
        with open(p_filename, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        # serialize json string in order to make it fit to a data frame
        if isinstance(json_data["data"], str):
            json_data["data"] = json.loads(json_data["data"])
        # convert json data to pandas DataFrame
        df = pd.DataFrame(json_data["data"])
        return df
 
   



