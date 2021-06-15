"""------------------------------------------------->
Description: Maintenance Model
    This maintenance model only takes into account
    bridges in Nebraska. Moreover, it takes into account
    bridges that have any repair, reconstruction, redeck,
    and overlay history within the NDOT spreadsheet provided
    by the NDOT.

Author: Akshay Kale
Date: June 9th, 2021
<------------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict

# System Libraries
import os
import sys

# ML
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

from decision_tree import *
from kmeans import *

def maintenance_pipeline(state):
    """
    Description:
        Pipeline for determining future maintenance of the bridges
    """

    # Creating directory
    csvfilename = state + '.csv'
    directory = state + 'OutputsTICR'

    # Create a state folder/ Change directory and then come out
    print("\n State: ", state)
    df = pd.read_csv(csvfilename, index_col=None, low_memory=False)

    # Change directory
    os.mkdir(directory)
    currentDir = os.getcwd()

    # Create results folders
    newDir = currentDir + '/' + directory
    os.chdir(newDir)
    modelOutput = state + 'ModelSummary.txt'

    sys.stdout = open(modelOutput, "w")
    print("\n state: ", state)
    resultsFolder = 'results'
    modelsFolder = 'models'
    os.mkdir(resultsFolder)
    os.mkdir(modelsFolder)

    # Remove null values
    df = df.dropna(subset=['deck',
                           'substructure',
                           'superstructure',
                           ]
                           )


    # Remove values encoded as N
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]

    # Fill the null values with -1
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)

#   Select columns for conversion and normalization
#   TODO: Consider putting number of intervention for deck, superstructure, and substructure
#   maintenance ( defined by the TICR ), and  features.
    columnsNormalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "avgDailyTruckTraffic",
                        "snowfall",
                        "freezethaw",
                        "futureNumberOfInterventions"
                      ]

    columnsFinal = [
                    "deck",
                    "yearBuilt",
                    "superstructure",
                    "substructure",
                    "averageDailyTraffic",
                    "avgDailyTruckTraffic",
                    "material",
                    "designLoad",
                    "snowfall",
                    "freezethaw",
                    "futureNumberOfInterventions",
                    "NdotIntervention"
                    ]

    dataScaled = normalize(df, columnsNormalize)
    dataScaled = dataScaled[columnsFinal]

    # Transform the dataset using k-means
    dataScaled = dataScaled[~dataScaled['NdotIntervention'].isin(
                                                                 ['REMOVE',
                                                                  'REDECK',
                                                                  'WIDEN - REDECK',
                                                                  'WIDEN - REHAB',
                                                                  'WIDEN - OVERLAY'])]
    counts = Counter(dataScaled['NdotIntervention'])
    # Remove columns
    columnsFinal.remove('NdotIntervention')

    # Modeling features and groundtruth:
    X, y = dataScaled[columnsFinal], dataScaled['NdotIntervention']

    # Oversampling
    oversample = SMOTE()
    print("\n Oversampling (SMOTE) ...")
    X, y = oversample.fit_resample(X, y)

    # Summarize distribution
    print("\n Distribution of the clusters after oversampling: ",
            Counter(y))

    # return to home directory
    kappaValues, accValues = decision_tree(X, y)
    sys.stdout.close()
    os.chdir(currentDir)

    return kappaValues, accValues

# Driver function
def main():
    # States
    filename = "nebraska"
    listOfKappaValues = list()
    listOfAccValues = list()

    # Output
    modelOutput = filename + 'ModelSummary.txt'
    kappa, acc = maintenance_pipeline(filename)
    listOfKappaValues.append(kappa)
    listOfAccValues.append(acc)

    sys.stdout = open("OverallOutput.txt", "w")

    #plot_overall_performance(csvfiles,
    #                         listOfKappaValues,
    #                         "kappaValues")

    #plot_overall_performance(csvfiles,
    #                         listOfAccValues,
    #                         "AccValues")

    sys.stdout.close()

if __name__=="__main__":
    main()
