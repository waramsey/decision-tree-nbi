"""--------------------------------------------------------------->
Description: Deterioration model
Author: Akshay Kale
Date: May 7th, 2021
<---------------------------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict

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
    csvfilename = state + '.csv'
    modelOutput = state + 'ModelSummary.txt'

    # Output
    sys.stdout = open(modelOutput, "w")

    # create a state folder/ Change directory and then come out
    print("\n State: ", state)
    df = pd.read_csv(csvfilename, index_col=None, low_memory=False)

    # Remove null values
    df = df.dropna(subset=['deck',
                           'substructure',
                           'superstructure',
                           "deckNumberIntervenion",
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
                        "deckNumberIntervenion"
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
                    "deckNumberIntervenion"
                    ]

    dataScaled = normalize(df, columnsNormalize)
    dataScaled = dataScaled[columnsFinal]

    # Create Categorization
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
                       'deckNumberIntervenion',
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
    counts = Counter(dataScaled['cluster'])
    numOfMembers = min(counts.values())

    # TODO: only if the number of members are low than 15
    indexes = list(counts.keys())
    vals = list(counts.values())
    minimum = vals.index(numOfMembers)
    minCluster = indexes[minimum]

    if numOfMembers < 15:
        print("\n Cluster with lowest membership (<15): ",
                minCluster, min(counts.values()))

    # Decision Tree
    columnsFinal.remove('deckDeteriorationScore')
    columnsFinal.remove('subDeteriorationScore')
    columnsFinal.remove('supDeteriorationScore')
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('deckNumberIntervenion')
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
    decision_tree(X, y)
    sys.stdout.close()

# Driver function
def main():

    csvfiles = ["nebraska",
                "nebraska"
                ]

    for filename in csvfiles:
        deterioration_pipeline(filename)

    #filename = 'nebraska.csv'
    #deterioration_pipeline(filename)

if __name__=="__main__":
    main()
