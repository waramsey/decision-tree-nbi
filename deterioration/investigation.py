"""
Description:
    Investigation in the High Deck - No Substructure - No Superstructure
"""
import csv
import pandas as pd
from collections import defaultdict

def main():
    df = pd.read_csv("structureNumbers.csv")
    neDf = pd.read_csv("nebraskaAll.csv")
    structureNumbers = list(df['structureNumber'])

    # Nebraska structure numbers
    neStructureNumbers = neDf['structureNumber']
    year = neDf['year']
    deck = neDf['deck']
    sub = neDf['substructure']
    sup = neDf['superstructure']

    # Create a dictionary 
    structDeck = defaultdict(list)
    structSub = defaultdict(list)
    structSup = defaultdict(list)
    structYear = defaultdict(list)

    for structureNumber, deck, sub, sup, year in zip(neStructureNumbers, deck, sub, sup, year):
        structDeck[structureNumber].append(deck)
        structSub[structureNumber].append(sub)
        structSup[structureNumber].append(sup)
        structYear[structureNumber].append(year)

    #for bridge in structureNumbers:
    #    print("\nbridge")
    #    print(bridge)
    #    print("\nDeck")
    #    print(list(zip(structYear[bridge], structDeck[bridge])))
    #    print("\nSubstructure")
    #    print(list(zip(structYear[bridge], structDeck[bridge])))
    #    print("\nSuperstructure")
    #    print(list(zip(structYear[bridge], structDeck[bridge])))

    bridge = 'C000104515'
    print("\nbridge")
    print(bridge)
    print("\nYear")
    print(structYear[bridge])
    print("\nDeck")
    print(structDeck[bridge])
    #print(list(zip(structYear[bridge], structDeck[bridge])))
    print("\nSubstructure")
    print(structSub[bridge])
    #print(list(zip(structYear[bridge], structDeck[bridge])))
    print("\nSuperstructure")
    print(structSup[bridge])
    #print(list(zip(structYear[bridge], structDeck[bridge])))


if __name__ =='__main__':
    main()
