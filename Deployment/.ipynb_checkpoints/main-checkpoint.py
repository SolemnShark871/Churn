#import the library
import streamlit as st
import pandas as pd
from helper import normalised_hist

#preprocessing

df_retired = pd.read_csv(r'C:\Users\dpanche\OneDrive - Asesoftware S.A.S\Documentos\Churn\static\data\retired_people.csv')
df_non_retired = pd.read_csv(r'C:\Users\dpanche\OneDrive - Asesoftware S.A.S\Documentos\Churn\static\data\non_retired_people.csv')

df_retired_nonretired = pd.concat([df_retired, df_non_retired], join = 'inner', ignore_index = True)
  
# Titulo
st.markdown('# Churn')
st.markdown("Entendiendo porque nuestros empleados se retiran de la empresa")

# Introducción
st.markdown("#### What is? ")
st.markdown("Churn is a ML based model capable to predict whether a determined employee will or not leave the company he or she works for in the next few months.")

st.markdown("#### How does it work?")
st.markdown("Our model has been trained through a series of data located on a database. These data in turn come from a daily activity report* software where the employee fill all the tasks he or she perform during the workdays.")

st.markdown("#### For what purpose?")
st.markdown("The purpose this predictive model was created for, lies in the help it provides to the companies to know earlier if an employee is planning to resign. By this way, there is more time for the companies to talk and to convince this person to stay.")

st.markdown("#### Who can benefit from this project?")
st.markdown("A wide variety of companies for which a considerable number of people work, avoiding them a considerable amount of hiring processes.")

st.markdown("#### Limitations?")
st.markdown("Currently, Churn is limited to those companies who have a daily activity report software* for their workers and have a register of the people who have left them")

st.markdown("#### How to use it?")
st.markdown("To get the most  out of this tool, just stick to the following steps:")
st.markdown("   1. Look for the data, the candidate you want to evaluate has filled on the daily activity report software")
st.markdown("   2. Take a window range period of a year and fetch the vector data of the candidate")
st.markdown("   3. Check that all the values are depicted into hour units.")
st.markdown("   4. Down, in the model section, put just the non zero values our candidate has in the data fetched.")
st.markdown("   5. Click on the model prediction button and wait a bit.")
st.markdown("   6. The model will give you the answer whether your candidate in thinking to leave or not the company")

# Datos

st.markdown('## Nuestros datos')

st.markdown("### Que datos usamos? ")

st.markdown("Se extrajeron datos de la plataforma ARES II que agrupamos en dos categorías")

st.markdown("#### Reporte diario de actividades")

st.markdown("* horas invertidas en actividades facturables o no facturables")
st.markdown("* horas dedicadas por cada empleado a trabajar en un área")
st.markdown("* actividades pendientes de cada empleado por área")
st.markdown("* horas invertidas trabajando en una tecnología especifica")

st.markdown("#### Caracteristicas de nuestros empleados")

st.markdown("* Meses trabajados en la empresa")
st.markdown("* Edad")

#Call the function
st.pyplot(normalised_hist(df_non_retired, df_retired, 'EDAD', 'Age', [10, 75]))

