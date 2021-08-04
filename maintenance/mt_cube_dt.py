"""------------------------------------------------->
Description: Maintenance Model
    This maintenance model only takes into account
    bridges all bridges in midwest.
Author: Akshay Kale
Date: August 9th, 2021
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
                           ])

    df = remove_duplicates(df)

    # Remove values encoded as N:
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]
    df = df[~df['scourCriticalBridges'].isin(['N', 'U', np.nan])]
    df = df[~df['deckStructureType'].isin(['N', 'U'])]

    # Fill the null values with -1:
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)
    df.toll.fillna(value=-1, inplace=True)
    df.designatedInspectionFrequency.fillna(value=-1, inplace=True)
    df.deckStructureType.fillna(value=-1, inplace=True)
    df.typeOfDesign.fillna(value=-1, inplace=True)

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
                        "deckNumberIntervention",

        # New
                        "latitude",
                        "longitude",
                        "skew",
                        "numberOfSpansInMainUnit",
                        "lengthOfMaximumSpan",
                        "structureLength",
                        "bridgeRoadwayWithCurbToCurb",
                        "operatingRating",
                        "scourCriticalBridges",
                        "lanesOnStructure",

                   #     "deckDeteriorationScore",
                   #     "subDeteriorationScore",
                   #     "supDeteriorationScore"
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
                    "latitude",
                    "longitude",
                    "skew",
                    "numberOfSpansInMainUnit",
                    "lengthOfMaximumSpan",
                    "structureLength",
                    "bridgeRoadwayWithCurbToCurb",
                    "operatingRating",
                    "scourCriticalBridges",
                    "lanesOnStructure",
                    "toll",
                    "designatedInspectionFrequency",
                    "deckStructureType",
                    "typeOfDesign",
                    "deckDeteriorationScore",
                    "subDeteriorationScore",
                    "supDeteriorationScore"
                ]


    dataScaled = normalize(df, columnsNormalize)
    dataScaled = dataScaled[columnsFinal]
    dataScaled = remove_null_values(dataScaled)

    # Data Scaled
    features = ["supNumberIntervention",
                "subNumberIntervention",
                "deckNumberIntervention"]

    sLabels = semantic_labeling(dataScaled[features], name="")
    dataScaled['cluster'] = sLabels
    print("\n Clusters:", Counter(list(dataScaled['cluster'])))

    # Analysis of Variance:
    #anovaTable, tukeys =  evaluate_ANOVA(dataScaled, features, lowestCount)
    #print("\nANOVA: \n", anovaTable)
    #print("\nTukey's : \n")
    #for result in tukeys:
    #    print(result)
    #    print('\n')

    # Characterizing the clusters:
    characterize_clusters(dataScaled, features)

    # Remove columns:
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('subNumberIntervention')
    columnsFinal.remove('deckNumberIntervention')

    labels = ['No Substructure - High Deck - No Superstructure',
             'High Substructure - No Deck - No Superstructure',
             'No Substructure - No Deck - High Superstructure']

    kappaValues = list()
    accValues = list()

    for label in labels:
        print("\nCategory: ", label)
        print("----------"*8)
        dataScaled = create_labels(dataScaled, label)
        clusters = Counter(dataScaled['label'])
        listOfClusters = list()
        for cluster in clusters.keys():
            numOfMembers = clusters[cluster]
            if numOfMembers < 15:
                listOfClusters.append(cluster)

        dataScaled = dataScaled[~dataScaled['label'].isin(listOfClusters)]

        # State column:
        dataScaled['state'] = [state]*len(dataScaled)

                # Modeling features and groundtruth:
        X, y = dataScaled[columnsFinal], dataScaled['label']

        # Oversampling:
        oversample = SMOTE()
        print("\n Oversampling (SMOTE) ...")
        X, y = oversample.fit_resample(X, y)

        # Summarize distribution:
        print("\n Distribution of the clusters after oversampling: ", Counter(y))

        # Return to home directory:
        kappaValue, accValue = decision_tree(X, y)
        kappaValues.append(kappaValue)
        accValues.append(accValue)
    sys.stdout.close()
    os.chdir(currentDir)

    return dataScaled, sLabels, kappaValues, accValues

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

    csvfiles = ['nebraska']
    listOfKappaValues = list()
    listOfAccValues = list()
    listOfLabels = list()
    listOfStates = list()
    listOfCounts = list()
    listOfDataFrames = list()

    for filename in csvfiles:
         filename = filename+'_deep'
         dataScaled, sLabel, kappaValues, accValues = maintenance_pipeline(filename)
         listOfLabels.append(sLabel)
         listOfStates.append(filename)
         listOfDataFrames.append(dataScaled)
         listOfKappaValues.append(kappaValues)
         listOfAccValues.append(accValues)

    sys.stdout = open("OverallOutput.txt", "w")

    ## Change the orientation:
    states = list()
    clusterNames = list()
    countsTemp = list()

    ## Print the values:
    for  sLabel, state, counts in zip(listOfLabels,
                                      listOfStates,
                                      listOfCounts):
        counts = dict(counts).values()
        for label, item1, count in zip(sLabel, state, counts):
            states.append(state)
            clusterNames.append(label)
            countsTemp.append(count)
    to_csv(listOfDataFrames)

    # TODO
    # For values in kappa values and accvalues
    # Edit kappaValues 
    #plot_overall_performance(csvfiles, # State values
    #                         listOfKappaValues,
    #                         "KappaValues",
    #                          state) # should be cluster value

    #plot_overall_performance(csvfiles,
    #                         listOfAccValues,
    #                         "AccValues"
    #                         state)

    sys.stdout.close()

if __name__=="__main__":
    main()
