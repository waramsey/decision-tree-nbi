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

#from decisionmethod import decision_tree
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

    # Remove null values:
    df = df.dropna(subset=['deck',
                           'substructure',
                           'superstructure',
                           'deckNumberIntervention',
                           'subNumberIntervention',
                           'supNumberIntervention',
                           ]
                           )

    df = remove_duplicates(df)

    # Remove values encoded as N:
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]

    # Fill the null values with -1:
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)

    # Normalize features:
    columnsNormalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "avgDailyTruckTraffic",
                        "supNumberIntervention",
                        "subNumberIntervention",
                        "deckNumberIntervention"
                      ]

    # Select final columns:
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
                    "supNumberIntervention",
                    "subNumberIntervention",
                    "deckNumberIntervention",
                    ]

    dataScaled = normalize(df, columnsNormalize)
    dataScaled = dataScaled[columnsFinal]

    # K-means:
    kmeans_kwargs = {
                        "init": "random",
                        "n_init": 10,
                        "max_iter": 300,
                        "random_state": 42,
                    }

    listOfParameters = ['supNumberIntervention',
                       'subNumberIntervention',
                       'deckNumberIntervention'
                      ]

    dataScaled, lowestCount, centroids = kmeans_clustering(dataScaled, listOfParameters, kmeans_kwargs, state=state)

    # Analysis of Variance:
    anovaTable =  evaluate_ANOVA(dataScaled, listOfParameters, lowestCount)
    print("\n ANOVA: \n", anovaTable)

    # Characterizing the clusters:
    characterize_clusters(dataScaled, listOfParameters)

    # Remove clusters with less than 15 members:
    clusters = Counter(dataScaled['cluster'])

    listOfClusters = list()
    for cluster in clusters.keys():
        numOfMembers = clusters[cluster]
        if numOfMembers < 15:
            listOfClusters.append(cluster)

    dataScaled = dataScaled[~dataScaled['cluster'].isin(listOfClusters)]

    # Remove columns:
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('subNumberIntervention')
    columnsFinal.remove('deckNumberIntervention')

    # Modeling features and groundtruth:
    X, y = dataScaled[columnsFinal], dataScaled['cluster']

    # Oversampling:
    oversample = SMOTE()
    print("\n Oversampling (SMOTE) ...")
    X, y = oversample.fit_resample(X, y)

    # Summarize distribution:
    print("\n Distribution of the clusters after oversampling: ",
            Counter(y))

    # Return to home directory:
    kappaValues, accValues = decision_tree(X, y)
    sys.stdout.close()
    os.chdir(currentDir)

    return kappaValues, accValues, centroids

# Driver function
def main():
    # States
    csvfiles = [
                "nebraska",
                "kansas",
                "indiana",
                "illinois",
                "ohio",
                "wisconsin",
                "missouri",
                "minnesota"
                ]

    csvfiles = ["nebraska"]
    listOfKappaValues = list()
    listOfAccValues = list()
    listOfCentroids = list()
    listOfStates = list()

    for filename in csvfiles:
         # Output
         kappa, acc, centroids = maintenance_pipeline(filename)
         listOfKappaValues.append(kappa)
         listOfAccValues.append(acc)
         listOfCentroids.append(centroids)
         listOfStates.append(filename)

    sys.stdout = open("OverallOutput.txt", "w")

    # Change the orientation:
    supNumberIntervention = list()
    deckNumberIntervention = list()
    subNumberIntervention = list()
    states = list()

    # Print the values:
    for cluster, state in zip(listOfCentroids, listOfStates):
        numOfItems = len(cluster)
        for item, item1 in zip(cluster, state):
            subNumberIntervention.append(item[0])
            deckNumberIntervention.append(item[1])
            supNumberIntervention.append(item[2])
            states.append(state)

    centroidDf = pd.DataFrame({'states': states,
                               'subNumInt': subNumberIntervention,
                               'deckNumInt': deckNumberIntervention,
                               'supNumInt': supNumberIntervention})

    print("\n Printing Centroids: ", centroidDf)

    plot_centroids(csvfiles,
                  centroidDf,
                  "Centroid")

    plot_overall_performance(csvfiles,
                             listOfKappaValues,
                             "KappaValues")

    plot_overall_performance(csvfiles,
                             listOfAccValues,
                             "AccValues")
    sys.stdout.close()

if __name__=="__main__":
    main()
