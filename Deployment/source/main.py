#---------------------------------------------- Libraries ----------------------------------------------
import streamlit as st
import pandas as pd
import os

#Own libraries
from helper import age_hist, time_company_hist

#---------------------------------------------- Preprocessing ----------------------------------------------
df_retired = pd.read_csv('../static/data/raw_data/retired_people.csv')
df_non_retired = pd.read_csv('../static/data/raw_data/non_retired_people.csv')
df_non_retired['LABEL'] = 'No retirado'
df_retired['LABEL'] = 'Retirado'
df_retired_nonretired = pd.concat([df_retired, df_non_retired], join = 'inner', ignore_index = True)

#---------------------------------------------- Page layout ----------------------------------------------
st.set_page_config(
    page_title = "Churn",
    page_icon = "../static/images/black-hole.png",
    layout = "centered"
)

# Title
#Layout variables
col01, col02 = st.columns([1, 2])

with col01:
    st.image("../static/images/black-hole.png", width = 200)
with col02:
    st.markdown("<h1 style = 'text-align: center; color: #f1c40f; font-size: 100px'>Churn</h1>", unsafe_allow_html = True)

st.markdown("""<p style = 'text-align: right; font-size: 20px; font-style: italic;'>
            Entendiendo porqué nuestros empleados se retiran de la empresa...</p>""", 
            unsafe_allow_html = True)
st.markdown("---")

#---------------------------------------------- Introduction section ----------------------------------------------
#Layout variables
col11, col12 = st.columns(2)

with col11:
    st.warning("#### ¿Qué es?")
    st.markdown("""Churn es un modelo de ML capaz de predecir si un determinado empleado dejará o no la 
                compañía para la que trabaja en los siguientes meses.""")
with col12:
    st.warning("#### ¿Cómo funciona?")
    st.markdown("""Nuestro modelo ha sido entrenado a través de una serie de datos localizados en una base de datos. Dichos datos a su vez provienen de un software de reporte de actividades 
                diarias* donde el empleado llena todas las tareas que ha realizado a lo largo de cada jornada laboral.""")

#Layout variables
col21, col22, col23 = st.columns(3)

with col21:
    st.warning("#### ¿Cuál es su propósito?")
    st.markdown("""El proposito para el cual este modelo fue creado, radica en la ayuda que éste provee a las compañías para saber de manera anticipada si un empleado está planeando 
                renunciar. De esta manera, las compañías tienen mas tiempo para hablar y convencer a esta persona de quedarse.""")
with col22:
    st.warning("#### ¿Quién puede beneficiarse de este proyecto?")
    st.markdown("""Una amplia variedad de compañías para las cuales 
                trabajen un número considerable de personas, evitando a estas una cantidad considerable de procesos de contratación.""")
with col23:
    st.warning("#### ¿Limitaciones?")
    st.markdown("""Actualmente, Churn se encuentra limitado a esas compañías que poseen un software de reporte de actividades
                diarias* para sus trabajadores y tienen además un registro de las personas que les han dejado.""")

st.warning("#### ¿Cómo usarlo?")
st.markdown("""Para sacar el mayor provecho de ésta herramienta, solo sigue los siguientes pasos:
* Busca los datos que el candidato a evaluar ha llenado en el software de reporte de actividades diarias*.
* Toma una ventana de tiempo de un año y extrae el vector con los datos del candidato.
* Revisa que todos los valores estén representados en horas.
* Debejo, en la sección correspondiente al modelo, pon solo los valores no nulos de nuestro candidato en la casilla correspondiente.
* Da click en el botón de predicción con nuestro modelo y espera un momento.
* El modelo arrojará su respuesta acerca de si el candidato está pensando o no en dejar la compañía.

**Ares II* en el caso específico de nuestra compañía: Asesoftware.""")

st.markdown("---")

#---------------------------------------------- Data section ----------------------------------------------
st.markdown("<h1 style = 'text-align: center; color: #76d7c4; font-size: 50px'>Nuestros datos</h1>", unsafe_allow_html = True)

st.info("#### ¿Qué datos usamos? ")
st.markdown("Se extrajeron datos de la plataforma *Ares II* (en el caso específico de nuestra compañía) que agrupamos en dos categorías principales:")

st.info("###### Reporte diario de actividades")

#Layout variables
col31, col32 = st.columns(2)

with col31:
    st.markdown("* Horas invertidas en actividades facturables o no facturables.")
    st.markdown("* Horas dedicadas por cada empleado a trabajar en un departamento especializado.")
with col32:
    st.markdown("* Actividades pendientes de cada empleado por departamento.")
    st.markdown("* Horas invertidas trabajando en una tecnología específica.")

st.info("###### Caracteristicas de nuestros empleados")
st.markdown("* Edad")
#Call the function
st.plotly_chart(age_hist(df_non_retired, df_retired), use_container_width = True)

st.markdown("* Meses trabajados en la empresa")
#Call the function
st.plotly_chart(time_company_hist(df_non_retired, df_retired), use_container_width = True)

#---------------------------------------------- Model section ----------------------------------------------
st.markdown("<h1 style = 'text-align: center; color: #7dcea0; font-size: 50px'>El modelo</h1>", unsafe_allow_html = True)
st.success("#### ¿Cómo se compone?")