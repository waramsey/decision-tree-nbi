"""
Description:
    Contains graphing functions

Author:
    Akshay Kale

Date: July 26, 2021
"""
import pandas as pd
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

def df_to_plotly(df):
    return {'z':df.values.tolist(),
            'x':df.columns.tolist(),
            'y':df.index.tolist()}

def plot_heatmap(df, title):
    dataHeatMap = df_to_plotly(df)
    print(dataHeatMap)
    fig = go.Figure(data=go.Heatmap(dataHeatMap))
    fig.update_layout(title_text=title,
                      font_size=15,
                      font=dict(size=15, color='black'),
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      barmode='group')
    savefile = title+'.html'
    fig.write_html(savefile)
    fig.show()

def plot_barchart(df, attribute, state, title):
    """
    Args:
        X: states
        Y: kappa or accuracy values
        names: clusters
    Returns:
        Plots
    """
    bars = list()
    savefile = title + '.html'
    clusters = df['cluster'].unique()
    for cluster in clusters:
        tempdf = df[df['cluster'].isin([cluster])]
        states = tempdf[state]
        vals =  tempdf[attribute]
        bars.append(go.Bar(name=cluster, x=states, y=vals))
    fig = go.Figure(data=bars)
    fig.update_layout(title_text=title,
                      font_size=15,
                      font=dict(size=15, color='black'),
                      xaxis=dict(title=state),
                      yaxis=dict(title=attribute),
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      barmode='group')
    fig.write_html(savefile)
    fig.show()

def plot_sankey_new(sources, targets, values, labels, title):
    """
    Description:
        Plots sankey diagram
    Args:
        sources (list)
        targets (list)
        values (list)
    Returns:
        plots
    """
    fig = go.Figure(data=[go.Sankey(
          node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),

          # Set of unique values
              label = labels,
              color = "blue"
        ),

        link = dict(
         #High Substructure
         source = sources,
         target = targets,
         value = values,

          #value = values,

          #color = ['plum']*6
          #      + ['powderblue']*6
          #      + ['purple']*6
          #      + ['red']*6
          #      + ['rosybrown']*6
          #      + ['royalblue']*6
          #      + ['rebeccapurple']*6
          #      + ['saddlebrown']*6

          #      + ['plum']*6
          #      + ['powderblue']*6
          #      + ['purple']*6
          #      + ['red']*6
          #      + ['rosybrown']*6
          #      + ['royalblue']*6
          #      + ['rebeccapurple']*6
          #      + ['saddlebrown']*6

          #      + ['plum']*6
          #      + ['powderblue']*6
          #      + ['purple']*6
          #      + ['red']*6
          #      + ['rosybrown']*6
          #      + ['royalblue']*6
          #      + ['rebeccapurple']*6
          #      + ['saddlebrown']*6
        #label = []
      ),
      )])
    fig.update_layout(title_text=title,
                      font_size=15,
                      font=dict(size=15, color='black'),
                      plot_bgcolor='white',
                      paper_bgcolor='white')
    fig.show()
    fig.write_html('important_features.html')
