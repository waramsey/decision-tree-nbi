"""
Description:
    This maintenance model only takes into account.

Author:
    Akshay Kale

Date:
    July 10, 2021
"""

# Systems
import os
import sys

# Data structures
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

from decision_tree import *
from kmeans import *

def utility_decision_tree(temp, state, category, modelOutput):
    """
    Description:
    Args:
    Returns:
    """
    #sys.stdout = open(modelOutput, "w")
    directory = category

    # Final columns
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

    # TODO: What the correct size of the membership?
    # Remove clusters with membership less than 15
    clusters = Counter(temp['cluster'])
    listOfClusters = list()
    for cluster in clusters.keys():
        numOfMembers = clusters[cluster]
        if numOfMembers < 15:
            listOfClusters.append(cluster)

    print(listOfClusters)
    temp = temp[~temp['cluster'].isin(listOfClusters)]
    # Remove columns
    columnsFinal.remove('deckDeteriorationScore')
    columnsFinal.remove('subDeteriorationScore')
    columnsFinal.remove('supDeteriorationScore')
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('subNumberIntervention')
    columnsFinal.remove('deckNumberIntervention')

    # Modeling features and groundtruth
    X, y =  temp[columnsFinal], temp['deterioration']

    # Print the distribution before oversampling
    print("\nDistribution before sampling:", Counter(y))

    # Oversampling
    print("\n Oversampling (SMOTE) .. ")
    oversampling = SMOTE()
    X, y =  oversampling.fit_resample(X, y)

    # Summarize 
    print("\nDistribution after sampling:", Counter(y))

    kappaValues, accValues = decision_tree(X, y)
    #os.chdir(currentDir)

    return kappaValues, accValues

def main():
    """
    Driver Function
    """
    df = pd.read_csv("allFiles.csv",
                    index_col=False)

    states = ['nebraska_deep',
              'kansas_deep',
              'indiana_deep',
              'illinois_deep',
              'ohio_deep',
              'wisconsin_deep',
              'missouri_deep',
              'minnesota_deep']

    #states = ['missouri_deep']
    # Substructure
    categories = categorize_attribute(df,
                                      'subDeteriorationScore',
                                      category=4)
    df['deterioration'] = categories
    uniqueCategories = df['cluster'].unique()
    currentDir = os.getcwd()
    modelOutput = 'clusterResults.csv'
    sys.stdout = open(modelOutput, "w")
    for category in uniqueCategories:
        # Creating directory for Category
        directory = currentDir +'/'+ category
        #os.mkdir(directory)
        #os.chdir(directory)

        # Creating temprorary dataframe
        tempCat = df[df['cluster'] == category]

        print("\n")
        print("Category: %s" % (category))
        print("----"*4)
        for state in states:
            temp = tempCat[tempCat['state'] == state]

            # Creating df state-category
            newDir = state
            #os.mkdir(newDir)
            #os.chdir(newDir)

            resultsFolder = 'results'
            modelsFolder = 'models'

            #os.mkdir(resultsFolder)
            #os.mkdir(modelsFolder)

            # Saving the model summary
            if len(temp) != 0:
                modelOutput = state + 'ModelSummary.txt'
                print('State: %s' %(state))
                #print("--------------------------")
                #print("Temporary Dataframe: ", temp.head())
                #print("\n")

                try:
                    utility_decision_tree(temp, state, category, modelOutput)
                except:
                    print("Can not build decision tree..")

            #os.chdir(directory)
        #os.chdir(currentDir)
    sys.stdout.close()


if __name__=='__main__':
    main()
