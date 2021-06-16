"""------------------------------------------------------>
Description: Deterioration model
Author: Akshay Kale
Date: May 7th, 2021
<------------------------------------------------------"""

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
from decision_tree import *
from kmeans import *

def deterioration_pipeline(state):
    """
    Description:
        Pipeline for deterioration
    """

    # Creating directory
    csvfilename = state + '.csv'
    directory = state + 'Outputs'

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
    print("\n State: ", state)
    resultsFolder = 'results'
    modelsFolder = 'models'
    os.mkdir(resultsFolder)
    os.mkdir(modelsFolder)

    # Remove null values
    df = df.dropna(subset=['deck',
                           'substructure',
                           'superstructure',
                           "deckNumberIntervention",
                           "subNumberIntervention",
                           "supNumberIntervention",
                           "subDeteriorationScore",
                           "supDeteriorationScore",
                           "deckDeteriorationScore"
                           ]
                           )

    #df = df.dropna(subset=['snowfall'])

    # The size of the dataframe
    df = remove_duplicates(df)

    # Remove values encoded as N
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]

    # Fill the null values with -1
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)

    # Select columns for conversion and normalization
    columnsNormalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "subDeteriorationScore",
                        "supDeteriorationScore",
                        "deckDeteriorationScore",
                        "supNumberIntervention",
                        "subNumberIntervention",
                        "deckNumberIntervention"
                      ]

    columnsFinal = [
                    "deck",
                    "yearBuilt",
                    "superstructure",
                    "substructure",
                    "averageDailyTraffic",
                    "material",
                    "designLoad",
                    "subDeteriorationScore",
                    "supDeteriorationScore",
                    "deckDeteriorationScore",
                    "supNumberIntervention",
                    "subNumberIntervention",
                    "deckNumberIntervention"
                    ]

    dataScaled = normalize(df, columnsNormalize)
    dataScaled = dataScaled[columnsFinal]

    # Create Categorization using normal distribution or K-means
    #dataScaled['label'] = categorize_attribute(dataScaled,
    #                                            'deteriorationScore')


    # K-means:
    ## Choose appropriate number of clusters
    kmeans_kwargs = {
                        "init": "random",
                        "n_init": 10,
                        "max_iter":300,
                        "random_state":42,
                    }

    listOfParameters = ['supDeteriorationScore',
                       'deckDeteriorationScore',
                       'subDeteriorationScore',
                       'supNumberIntervention',
                       'deckNumberIntervention',
                       'subNumberIntervention']

    # K-means clustering
    # Change in directory
    dataScaled, lowestCount = kmeans_clustering(dataScaled, listOfParameters, kmeans_kwargs)

    # Analysis of variance
    anovaTable = evaluate_ANOVA(dataScaled, listOfParameters, lowestCount)
    print("\n ANOVA: \n", anovaTable)

    # Analysis of the clusters:
    # Change in directory
    characterize_clusters(dataScaled, listOfParameters)

    # Transform the dataset
    #X, y = dataScaled[columnsFinal], dataScaled['label']

    # Remove cluster with less than 15 members:
    counts = Counter(dataScaled['cluster'])
    numOfMembers = min(counts.values())

    indexes = list(counts.keys())
    vals = list(counts.values())
    minimum = vals.index(numOfMembers)
    minCluster = indexes[minimum]

    if numOfMembers < 15:
        print("\n Cluster with lowest membership (<15): ",
                minCluster)
        print("\n Number of members in the cluster:", min(counts.values()))

    # Decision Tree
    columnsFinal.remove('deckDeteriorationScore')
    columnsFinal.remove('subDeteriorationScore')
    columnsFinal.remove('supDeteriorationScore')
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('deckNumberIntervention')
    columnsFinal.remove('subNumberIntervention')

    dataScaled = dataScaled[dataScaled['cluster'] != minCluster]
    X, y = dataScaled[columnsFinal], dataScaled['cluster']

    # Oversampling
    oversample = SMOTE()
    print("\n Oversampling (SMOTE) ...")
    X, y = oversample.fit_resample(X, y)

    # Summarize distribution
    print("\n Distribution of the clusters after oversampling: ",
            Counter(y))

    # Change in directory
    kappaVals, accVals  = decision_tree(X, y)

    # Work here:
    sys.stdout.close()
    os.chdir(currentDir)

    return kappaVals, accVals

# Driver function
def main():
    # States
    csvfiles = [
                "nebraska",
                "kansas",
                "indiana",
                "illinois",
                "ohio",
                #"northdakota", #[X]
                "wisconsin",
                "missouri",
                "minnesota",
                #"michigan" # [X]
                ]

    listOfKappaValues = list()
    listOfAccValues = list()
    for filename in csvfiles:
        # Output
        kappa, acc = deterioration_pipeline(filename)
        listOfKappaValues.append(kappa)
        listOfAccValues.append(acc)

    sys.stdout = open("OverallOutput.txt", "w")
    plot_overall_performance(csvfiles,
                             listOfKappaValues,
                             "KappaValues")

    plot_overall_performance(csvfiles,
                             listOfAccValues,
                             "AccValues")
    sys.stdout.close()

if __name__=="__main__":
    main()
