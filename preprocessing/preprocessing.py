"""
NOT BEING USED
DONOT NEED IT

description: functions to process NBE data
    features include:
        1. Creating structure number: NBE feature map


author: Akshay Kale
"""

import csv
from collections import defaultdict
from collections import namedtuple


from maps import *
from nbi_data_chef import *

def read_csv(filename):
    """
    Reads a csv file and returns a list of records
    """
    listOfNbeRecords = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        header = next(csvReader)
        if header[0] == '':
            header[0] = 'id'
        header = [col.replace(" ", "") for col in header]
        header = [col.replace("#", "No") for col in header]
        header = [col.replace("(mm)", "") for col in header]
        header = [col.replace(".", "") for col in header]
        Record = namedtuple("Record", header)
        for row in tqdm(csvReader, desc="Reading file"):
            record = Record(*row)
            listOfNbeRecords.append(record)
    return listOfNbeRecords, header

def generate_map(listOfNbeRecords, value, key=''):
    """
    Description:
    Args:
    Returns:
    """
    pass

def process_nbe():
    """
    Description: given the csvfile, create a dictionary of
    structure number and nbe dataset
    """
    cheatFileName = '/home/akshay/Documents/github/data/nbe_processed/2015-2019_nbe.csv'
    structNbeMap = defaultdict()
    listOfNbeRecords, header = read_csv(cheatFileName)
    return structNbeMap

if __name__ =='__main__':
    process_nbe()
