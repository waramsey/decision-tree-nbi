"""--------------------------------------------------------------------->
Description: Deterioration model
Author: Akshay Kale
Date: May 7th, 2021
<---------------------------------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict


# ML
from imblearn.over_sampling import SMOTE
from decision_tree import *
from kmeans import *

# Driver function
def main():
    # CSVFile
    csvfilename = "nebraska.csv"
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

    # Decision Tree
    columnsFinal.remove('deckDeteriorationScore')
    columnsFinal.remove('subDeteriorationScore')
    columnsFinal.remove('supDeteriorationScore')
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('deckNumberIntervenion')
    columnsFinal.remove('subNumberIntervention')

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

    listOfParameter = ['supDeteriorationScore', 'deckDeteriorationScore', 'subDeteriorationScore', 'supNumberIntervention', 'deckNumberIntervenion', 'subNumberIntervention']
    dataScaled = kmeans_clustering(dataScaled, listOfParameter, kmeans_kwargs)

    # Transform the dataset
    #X, y = dataScaled[columnsFinal], dataScaled['label']
    counts = Counter(dataScaled['cluster'])
    indexes = list(counts.keys())
    vals = list(counts.values())
    minimum = vals.index(min(counts.values()))
    minCluster = indexes[minimum]
    print("\n Minimum Cluster: ", minCluster)

    dataScaled = dataScaled[dataScaled['cluster'] != minCluster]
    X, y = dataScaled[columnsFinal], dataScaled['cluster']


    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    # Summarize distribution
    print(Counter(y))

    decision_tree(X, y)

if __name__=="__main__":
    main()