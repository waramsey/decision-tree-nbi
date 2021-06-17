"""
description: functions to process precipitation data
    features include:
        1. Creates structure number -> precipitation map
        2. Population and feteching of precipitation data into mongodb
author: Akshay Kale
"""
import csv
from collections import defaultdict
from collections import namedtuple

from maps import *
from nbi_data_chef import *

# TOOD:
    # Make a procedure to store precipitation files into the MongoDB database
def read_csv(filename):
    listOfPrecip = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        header = next(csvReader)
        if header[0] ==  '':
            header[0] = 'id'
        header = [col.replace(" ", "") for col in header]
        header = [col.replace("#", "No") for col in header]
        header = [col.replace("(mm)", "") for col in header]
        header = [col.replace(".", "") for col in header]
        Record = namedtuple('Record', header)
        for row in tqdm(csvReader, desc='Reading file'):
            record = Record(*row)
            listOfPrecip.append(record)
    return listOfPrecip, header

def reverse_dict(dictionary):
    newDict = defaultdict()
    for key, value in zip(dictionary.keys(), dictionary.values()):
        newDict[value] = key
    return newDict

def make_fips_precp(listOfPrecip, state_code_mapping):
    # query mongodb
    fips_precp_dict = defaultdict()
    #for record in listOfPrecip:
    return fips_precp_dict

def query_db():
    # define query
    fields ={"_id":0,
             "structureNumber":1,
             "countyCode":1,
             "stateCode":1
            }

    years = [2016]
    #states = code_state_mapping.keys()
    states=[31]
    nbiDB = get_db()
    collection = nbiDB['nbi']
    #print(collection)
    results = query(fields, years, states, collection)
    #print(results)
    return results


def generate_map(listOfRecords, value, key='StructureNumber'):
    """
    description:
       generates a map of structure number and the selected value
    args:
        listofrecords (list of named tuple):
        value (string):
        key (string):
    returns:
        generatedmap (dictionary):
    """
    generatedMap = defaultdict()
    for record in listOfRecords:
        fieldValue = record._asdict()[value]
        keyValue = record._asdict()[key]
        keyValue = keyValue[:-3].strip(" ")
        generatedMap[keyValue] = fieldValue
    return generatedMap

def process_precipitation():
    # TODO: 
    # Create a function to insert precipitation data into mongodb
    # Create a function to query precipitation data from mongodb
    # Cheat Code: use precipitation, snowfall, and freeze thaw from already preprocessed.
    #filename = 'precipitation-allstates-2011.csv'
    #state_code_mapping = reverse_dict(code_state_mapping)
    #listOfPrecip, header= read_csv(filename)
    #records = query_db()

    #### process data through a cheat file ####
    cheatFilename = '06-20-19-thesis-dataset_allstates_allstates.csv'
    listOfPrecips, cheatHeader= read_csv(cheatFilename)
    structBdsMap = generate_map(listOfPrecips, 'BaselineDifferenceScore')
    structPrecipMap = generate_map(listOfPrecips, 'AvgDailyPrecipitation')
    return structBdsMap, structPrecipMap

if __name__ =='__main__':
    process_precipitation()

