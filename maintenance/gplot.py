"""
Description:
    Contains function to plotting functions

Author:
    Akshay Kale

Date: July 26, 2021
"""

import plotly.express as px

def plot_us():
    df = px.data.election()
    geojson = px.data.election_geojson()

    fig = px.choropleth(df, geojson=geojson,
                        color="Bergeron",
                        locations="district",
                        featureidkey="properties.district",
                        projection="mercator")

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

# Driver Function
def main():
   plot_us()

if __name__=='__main__':
    main()

