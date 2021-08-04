"""
Description:
    Contains function to plotting functions

Author:
    Akshay Kale

Date: July 26, 2021
"""
import plotly.graph_objects as go
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

def plot_sankey(title):
    """
    Description:
    Args:
    Return:
    """
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          # Set of unique values
          label = ["Nebraska",
                   "Illinois",
                   "Indiana",
                   "Missouri",
                   "Ohio",
                   "Wisconsin",
                   "Minnesota",
                   "Kansas",

                   "High Substructure",
                   "High Deck",
                   "High Superstructure",

                   "Deck Rating",
                   "Substructure Rating",
                   "Superstructure Rating",

                   "Latitude",
                   "Longitude",

                   "Substructure Slope",
                   "Superstructure Slope",
                   "Deck Slope",

                   "Year",
                   "Deck Structure Type",
                   "Design Load",
                   "Average Daily Traffic"
                   "Lanes on Sturcture",
                   "Bridge Roadway Curb to Curb",
                   "Length of Maximum Span"

                  ],

          color = "blue"
        ),
        link = dict(
            #High Substructure
          source = [0, 8, 8, 8, 8, 8,
                    1, 8, 8, 8, 8, 8,
                    2, 8, 8, 8, 8, 8,
                    3, 8, 8, 8, 8, 8,
                    4, 8, 8, 8, 8, 8,
                    5, 8, 8, 8, 8, 8,
                    6, 8, 8, 8, 8, 8,
                    7, 8, 8, 8, 8, 8,

            # High Deck
                    0, 9, 9, 9, 9, 9,
                    1, 9, 9, 9, 9, 9,
                    2, 9, 9, 9, 9, 9,
                    3, 9, 9, 9, 9, 9,
                    4, 9, 9, 9, 9, 9,
                    5, 9, 9, 9, 9, 9,
                    6, 9, 9, 9, 9, 9,
                    7, 9, 9, 9, 9, 9,

            # High Superstructure
                    0, 10, 10, 10, 10, 10,
                    1, 10, 10, 10, 10, 10,
                    2, 10, 10, 10, 10, 10,
                    3, 10, 10, 10, 10, 10,
                    4, 10, 10, 10, 10, 10,
                    5, 10, 10, 10, 10, 10,
                    6, 10, 10, 10, 10, 10,
                    7, 10, 10, 10, 10, 10,
                   ],

          target = [8, 16, 11, 12, 13, 14,
                    8, 12, 11, 13, 15, 14,
                    8, 12, 11, 14, 16, 15,
                    8, 12, 11, 14, 16, 15,
                    8, 16, 12, 11, 13, 18,
                    8, 16, 12, 11, 13, 18,
                    8, 12, 11, 13, 18, 15,
                    8, 11, 12, 14, 13, 16,

            # High Deck
                    9, 12, 13, 18, 16, 11,  # Nebraska
                    9, 12, 11, 13, 21, 14,  # Illinois
                    9, 13, 11, 12, 14, 17,  # Indiana
                    9, 12, 11, 20, 14, 16,  # Missouri
                    9, 12, 13, 18, 17, 11,  # Ohio
                    9, 12, 11, 13, 17, 18,  # Wisconsin
                    9, 11, 13, 12, 19, 14,  # Minnesota
                    9, 11, 13, 12, 15, 14,  # Kansas

            # High Superstructure
                    10, 13, 11, 12, 17, 18, # Nebraska 14
                    10, 13, 11, 18, 12, 15, # Illinois
                    10, 17, 13, 11, 12, 18, # Indiana
                    10, 12, 13, 16, 11, 22, # Missouri
                    10, 13, 11, 17, 18, 23, # Ohio
                    10, 17, 11, 13, 18, 24, # Wisconsin
                    10, 13, 17, 25, 12, 15, # Minnesota
                    10, 13, 11, 12, 17, 14, # Kansas
                   ],

          value =  [1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,

                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,

                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1,
                    1, 5, 4, 3, 2, 1],

         color = ['plum']*6
                + ['powderblue']*6
                + ['purple']*6
                + ['red']*6
                + ['rosybrown']*6
                + ['royalblue']*6
                + ['rebeccapurple']*6
                + ['saddlebrown']*6

                + ['plum']*6
                + ['powderblue']*6
                + ['purple']*6
                + ['red']*6
                + ['rosybrown']*6
                + ['royalblue']*6
                + ['rebeccapurple']*6
                + ['saddlebrown']*6

                + ['plum']*6
                + ['powderblue']*6
                + ['purple']*6
                + ['red']*6
                + ['rosybrown']*6
                + ['royalblue']*6
                + ['rebeccapurple']*6
                + ['saddlebrown']*6

        #label = []
      ))])
    fig.update_layout(title_text=title, font_size=15)
    fig.show()

# Driver Function
def main():
    title = 'Respresentation of important variables'
    plot_sankey(title)

if __name__=='__main__':
    main()

