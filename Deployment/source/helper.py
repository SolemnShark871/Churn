#Libraries
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff

# Sets the histogram for the age of the dataframes which are as parameters
def age_hist(df_nr, df_r):
    #Erasing ages below 16 years
    df_nr = df_nr[df_nr['EDAD'] > 16]
    df_r = df_r[df_r['EDAD'] > 16]
    #Concatenating both dataframes into one
    df_merged = pd.concat([df_r, df_nr], join = 'inner', ignore_index = True)
    #Histogram
    labels = {'EDAD': 'Edad (Años)', 'LABEL': 'Estado del empleado'}
    fig = px.histogram(df_merged, x = 'EDAD', color = 'LABEL', labels = labels, marginal = 'box')
    fig.update_yaxes(title_text = 'Frecuencia')
    fig.update_layout(bargap = 0.1, title_font_size = 20, title_x = 0.5)
    #Normal distribution

    return fig

# Sets the histogram for the time in the company of the dataframes which are as parameters
def time_company_hist(df_nr, df_r):
    #Erasing values above 0 months
    df_nr = df_nr[df_nr['TIEMPO_EMP'] > 0]
    df_r = df_r[df_r['TIEMPO_EMP'] > 0]
    #Concatenating both dataframes into one
    df_merged = pd.concat([df_r, df_nr], join = 'inner', ignore_index = True)
    #Histogram
    labels = {'TIEMPO_EMP': 'Tiempo en la empresa (Meses)', 'LABEL': 'Estado del empleado'}
    fig = px.histogram(df_merged, x = 'TIEMPO_EMP', color = 'LABEL', labels = labels, marginal = 'box')
    fig.update_yaxes(title_text = 'Frecuencia')
    fig.update_layout(bargap = 0.1, title_font_size = 20, title_x = 0.5)
    #Exponential distribution

    return fig