"""--------------------------------------------------------------------->
Description: preprocessing library
Author: Akshay Kale
Date: May 7th, 2021
<---------------------------------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from tqdm import tqdm

#  Metrics and stats
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Function for normalizing
def normalize(df, columns):
    """
    Function for normalizing the data
    """
    for feature in columns:
        df[feature] = df[feature].astype(int)
        maxValue = df[feature].max()
        minValue = df[feature].min()
        df[feature] = (df[feature] - minValue) / (maxValue - minValue)
    return df

# Function to group by
def drop_duplicate_rows(df):
    """
    Function to groupby bridge records
    """
    df = df.drop_duplicates(subset=['structureNumber'], keep='first', inplace=True)
    return df

# Differences in between the clusters 
def ANOVA(df, feature):
    """
    Description:
        Takes in dataframe with three columns, each column
        presents the value of an attribute from each cluster.

    Args:
        df (dataframe)
        feature (list)

    Returns:
        anovaTable (dataframe)
    """
    dfMelt = pd.melt(df.reset_index(), id_vars=['index'])
    dfMelt.columns = ['index', 'cluster', 'value']
    plot_box(dfMelt)
    model = ols('value ~ C(cluster)', data=dfMelt).fit()
    anovaTable = sm.stats.anova_lm(model, typ=2)
    return anovaTable

def plot_elbow(sse):
    """
    Description:
        Plot the elbow to find the optimal number of
    clusters
    """
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Elbow")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig("Elbow.png")

def plot_silhouette(silhouette):
    """
    Plot silhouette constant for finding optimal number
    of clusters
    """
    plt.style.use("fivethirtyeight")
    plt.title("Plotting Silhouette")
    plt.plot(range(2, 11), silhouette)
    plt.xticks(range(2, 12))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette coefficient")
    plt.savefig("Silhouette.png")

def plot_scatter(col1, col2, color=None):
    """
    Scatter plot for maximum number of bridges
    """
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Baseline Difference Score Vs. Deterioration Score ")
    plt.scatter(col2, col1)
    plt.xlabel("BaselineDiferenceScore")
    plt.ylabel("DeteriorationScore")
    plt.savefig("Scatter.png")

# Plot Scatter plot
def plot_scatter_Kmeans(df):
    """
    Description:
        Scatter plot for displaying k-means clusters
    """
    cluster1 = df[df['cluster'] == 0]
    cluster2 = df[df['cluster'] == 1]
    cluster3 = df[df['cluster'] == 2]

    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Plotting Clusters")
    plt.scatter(cluster1['deck'], cluster1['superstructure'], color='red')
    plt.scatter(cluster2['deck'], cluster2['superstructure'], color='green')
    plt.scatter(cluster3['deck'], cluster3['superstructure'], color='blue')
    plt.xlabel("Column 1")
    plt.ylabel("Column 2")
    plt.savefig("kmeans.png")

# Plot boxplot
def plot_box(dfMelt):
    """
    Description:
        Scatter plot for distribution bridge features
    """
    filename = "results/" + "KmeansFeatureBoxplot.png"
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Box plot")
    sns.boxplot(x='cluster', y='value', data=dfMelt, color='#99c2a2')
    plt.savefig(filename)

# Confusion matrix
def print_confusion_matrix(cm):
    """
    Description:
        Plots confusion matrix on validation set
    """
    indexList = list()
    columnList = list()

    for row in range(0, np.shape(cm)[0]):
        indexString = 'Actual' + str(row+1)
        columnString = 'Predicted' + str(row+1)
        indexList.append(indexString)
        columnList.append(columnString)

    dfCm = pd.DataFrame(cm,
                        index=indexList,
                        columns=columnList
                        )

    plt.subplots(figsize=(8, 8))
    sns.heatmap(dfCm, annot=True, fmt='g', cmap='BuPu')
    plt.savefig("ConfusionMatrix.png")

# Logistic Regression
def log_regression(df, X, y):
    """
    Description:
        Performs training testing split
        Perform multi-nomial logistic regression

    Args:
        df (dataframe)

    Returns:
        model (sklearn model)
    """
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)
    model = LogisticRegression(multi_class="multinomial", random_state=0)
    model.fit(trainX, trainy)
    prediction = model.predict(testX)
    cm = confusion_matrix(testy, prediction)
    cr = classification_report(testy, prediction)
    r2 = r2_score(testy, prediction)
    print("\n Intercept: \n", model.intercept_)
    print("\n Coef: \n", model.coef_)
    print("\n Classes: ", model.classes_)
    print("\n Classification: \n", cr)
    print("\n R Square: ", r2)
    print("\n Confusion matrix: \n", cm)
    print_confusion_matrix(cm)
    return model

def evaluate_ANOVA(dataScaled, columns, lowestCount):
    """
    Description:
        Compute ANOVA table for all features in all
        Clusters. And, returns ANOVA to find essential features
        in the cluster analysis

    Args:
        dataScaled (dataframe)
        columns (list)
        lowestCount (integer)

    Returns:
        anovaDf (dataframe)
    """
    pvalues = list()
    features = list()
    sumsqs = list()
    dfs = list()
    fvalues = list()

    for col in columns:
        temp = defaultdict()
        for rows in dataScaled.groupby('cluster')[col]:
            cluster, records = rows
            temp[cluster] = np.random.choice(list(records), lowestCount)
        tempDf = pd.DataFrame.from_dict(temp)
        anovaTable = ANOVA(tempDf, col)

        pvalue = anovaTable['PR(>F)'][0]
        pvalues.append(pvalue)

        sumsq = anovaTable['sum_sq'][0]
        sumsqs.append(sumsq)

        df = anovaTable['df'][0]
        dfs.append(df)

        F = anovaTable['F'][0]
        fvalues.append(F)

        features.append(col)

    anovaDf = pd.DataFrame(columns=['Attribute',
                                    'sum_sq',
                                    'df',
                                    'F',
                                    'p-value'])
    anovaDf['Attribute'] = features
    anovaDf['sum_sq'] = sumsqs
    anovaDf['df'] = dfs
    anovaDf['F'] = fvalues
    anovaDf['p-value'] = pvalues
    return anovaDf

def kmeans_clustering(dataScaled, listOfParameters, kmeans_kwargs):
    """
    Description:
        Performs selection of optimal clusters and kmeans
        and returns an updated dataframe with cluster membership
        for each bridge

    Args:
       dataScaled (dataframe)
       listOfParameters (dataframe)
       kmeans_kwargs (dictionary)

    Returns:
        dataScaled (dataframe)
    """
    sse = list()
    for k in tqdm(range(1, 11), desc='Computing Elbow'):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(dataScaled[listOfParameters])
        sse.append(kmeans.inertia_)

    # Plot SSE
    plot_elbow(sse)

    # Find optimum number of clusters
    kl = KneeLocator(range(1, 11),
                    sse,
                    curve='convex',
                    direction='decreasing')

    # Print elbow and silhouette coefficient
    print("\n Optimal number of cluster (Elbow Coefficient): ", kl.elbow)

    # Use K-means with optimal numbe of cluster
    finalKmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs)
    finalKmeans.fit(dataScaled[listOfParameters])

    counts = Counter(finalKmeans.labels_)
    lowestCount = min(counts.values())

    print("\n Number of members in cluster:",
            counts)

    # Save cluster as columns
    dataScaled['cluster'] = list(finalKmeans.labels_)
    return dataScaled, lowestCount

# Driver function
def main():

    # CSVFile
    csvfilename = "nebraska_data.csv"
    df = pd.read_csv(csvfilename, index_col=None, low_memory=False)

    # Remove null values
    df = df.dropna(subset=['deck',
                           'substructure',
                           'superstructure']
                           )

    df = df.dropna(subset=['snowfall'])

    # The size of the dataframe
    print("\n Size of the dataframe: " , np.size(df))
    rows = np.size(df)

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
    columnsFinal = [
                    "deck",
                    "yearBuilt",
                    "superstructure",
                    "substructure",
                    "averageDailyTraffic",
                    "avgDailyTruckTraffic",
                    "deteriorationScore",
                    "material",
                    "designLoad",
                    "wearingSurface",
                    "structureType"
                    ]

    dataScaled = normalize(df, columnsFinal)
    dataScaled = dataScaled[columnsFinal]

    # Plotting score
    plot_scatter(dataScaled['superstructure'], dataScaled['deck'])

    # Choosing appropriate number of clusters
    kmeans_kwargs = {
                         "init":"random",
                         "n_init": 10,
                         "max_iter": 300,
                         "random_state": 42,
                    }

    # Elbow method
    listOfParameters = ['deteriorationScore', 'superstructure']
    dataScaled, lowestCount = kmeans_clustering(dataScaled, listOfParameters, kmeans_kwargs)

    # Analysis of variance
    anovaTable = evaluate_ANOVA(dataScaled, columnsFinal, lowestCount)
    print("\n ANOVA: \n", anovaTable)

    # Plot Clusters
    plot_scatter_Kmeans(dataScaled)

    # Multinomal Logistic Regression
    columnsFinal.remove('superstructure')
    columnsFinal.remove('substructure')
    columnsFinal.remove('deck')
    X, y = dataScaled[columnsFinal], dataScaled['cluster']
    log_regression(dataScaled, X, y)

if __name__=="__main__":
    main()
