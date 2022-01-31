#---------------------------------------------- Libraries ----------------------------------------------
from os import path
import pandas as pd
import numpy as np
import datetime
from datetime import date
import config
import time

#---------------------------------------------- Class to make some inferences ----------------------------------------------
# Class that makes the inferences from the datasets through the ML models alrady trained
class Inference():
    def __init__(self) -> None:
        pass

    # Function to inport the datasets with the data on which predictions are made
    # dict_r, dict_nr: diccionaries with the SQL queries and the functions to fetch the data from the database
    def inport_data(self, dict_r: dict, dict_nr: dict):
        pass


#---------------------------------------------- Calls ----------------------------------------------

