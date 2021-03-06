#---------------------------------------------- Libraries ----------------------------------------------
import pandas as pd
import numpy as np
import datetime
from datetime import date
import cx_Oracle
import config
import time
from azure.storage.blob import BlobServiceClient
import json



#---------------------------------------------- Class to extract the data ----------------------------------------------
# The class connects to the database and extracts the data that will be used to ML purposes
class Connection():
	'''Conexión a la base de datos'''
	#cx_Oracle.init_oracle_client(lib_dir=config.lib_dir)
	def __init__(self):
		self.connection = None

	# Makes the connection to the database, if not possible throws an error
	def try_connection(self):
		try:
			print("username:",config.username)
			self.connection = cx_Oracle.connect(
				config.username,
				config.password,
				config.dsn,
				encoding = config.encoding)
			# Show the version of the Oracle Database
			print("conection_version:",self.connection.version)
		except cx_Oracle.Error as error:
			print("error:", error)
				
	# Function to calculate people's age
	# in_date: date value to calculate a period range referred to the age
	def age_f(self, in_date) -> int:
		if len(in_date) == 1:
			self.last_date = date.today() #Y M D
		else:
			self.last_date = in_date[1]

		self.birth_date = in_date[0]
		age = (self.last_date.year - self.birth_date.year - ((self.last_date.month - 2, self.last_date.day) < 
				(self.birth_date.month, self.birth_date.day)))
		return age

	# Function to calculate people's time in the company
	# in_date: date value to calculate a period range referred to the time in the company
	def time_f(self, in_date) -> int:
		if len(in_date) == 1:
			self.last_date = date.today() #Y M D
		else:
			self.last_date = in_date[1]
		
		self.enter_date = in_date[0]
		months = (self.last_date.year - self.enter_date.year)*12 + (self.last_date.month - self.enter_date.month) - 2
		return months

    # Function to query the data to the database through SQL strings
	# dictionary: is the container of the SQl queries together with the functions they need to be fetched
	def make_query (self, dictionary: dict) -> dict:
		self.cur = self.connection.cursor()
		time_start = datetime.datetime.now()
		dic_df = {}
		for key in dictionary:
			query = dictionary[key][0]
			list_title_query = dictionary[key][1]
			list_functions = dictionary[key][2]

			df_str = f"df_{list_title_query[0]}"
			
			if "no_function" in list_functions:
					self.cur.execute(query)
					self.res = self.cur.fetchall()
					# Create the dataframe
					dic_df[df_str]= pd.DataFrame(self.res, columns = ['PERSONA', list_title_query[0]])
				
			elif "pivot_table" in list_functions:
				self.cur.execute(query)
				self.res = self.cur.fetchall()
				# Create the dataframe
				dic_df[df_str] = pd.DataFrame(self.res, columns = ['PERSONA', list_title_query[1], list_title_query[0]])
					
				if len(list_functions) == 1:
					self.cur.execute(query)
					self.res = self.cur.fetchall()
					# Create the dataframe
					dic_df[df_str] = pd.pivot_table(dic_df[df_str], values = list_title_query[0], index = ['PERSONA'], columns = list_title_query[1], fill_value = 0)

				else:
					if "suffix" in list_functions:
						self.cur.execute(query)
						self.res = self.cur.fetchall()
						# Additional data managing
						dic_df[df_str] = pd.pivot_table(dic_df[df_str], values = list_title_query[0], index = ['PERSONA'], columns = list_title_query[1], fill_value = 0)
						dic_df[df_str].columns = [str(col) + f"_{list_functions[-1]}" for col in dic_df[df_str].columns]
					
					elif "fillna" in list_functions:
						self.cur.execute(query)
						self.res = self.cur.fetchall()
						# Additional data managing
						dic_df[df_str][list_title_query[1]] = dic_df[df_str][list_title_query[1]].fillna(list_functions[-1])
						dic_df[df_str] = pd.pivot_table(dic_df[df_str], values = list_title_query[0], index = ['PERSONA'], columns = list_title_query[1], fill_value = 0)

			elif ("age" in list_functions) or ("time" in list_functions):
				self.cur.execute(query)
				self.res = self.cur.fetchall()
				func = self.age_f if "age" in list_functions else self.time_f
				# List with the query columns needed to the apply function
				list_columns = list_title_query[1:]
				list_columns.insert(0, 'PERSONA')
				# Create the dataframe
				dic_df[df_str] = pd.DataFrame(self.res, columns = list_columns)
				dic_df[df_str][list_title_query[0]] = dic_df[df_str][list_title_query[1:]].apply(func, axis = 1)
				dic_df[df_str].drop(columns = list_title_query[1:], inplace = True)

			elif "replace" in list_functions:
				self.cur.execute(query)
				self.res = self.cur.fetchall()
				# Create the dataframe
				dic_df[df_str] = pd.DataFrame(self.res, columns = ['PERSONA', list_title_query[0]])
				# 0 for Men and 1 for Women
				dic_df[df_str].replace(to_replace = ['M', 'F'], value = [0, 1], inplace = True)

		for k in dic_df:
			print(k)
                    
		query_time = datetime.datetime.now() - time_start
		print(f"Duración de Consulta (seg): {query_time}")

		return dic_df

	# Function to merge the dataframes which will made up the dataset
	# dictionary: is the container of the dataframes with keys as the subsets fetched from the database
	def merge_df(self, dictionary: dict) -> pd.DataFrame:
		time_start = datetime.datetime.now()
		keys = list(dictionary.keys())
		df = dictionary[keys[0]]
		for i in range(1, len(keys)):	
			df = pd.merge(df, dictionary[keys[i]], left_on = 'PERSONA', right_on = 'PERSONA')

		query_time = datetime.datetime.now() - time_start
		print(f"Duración del procedimiento (seg): {query_time}")

		return df

	# Function to export the dataframes from the database as csv files
	# path: the direction where to find the file
	# in_df: the dataframe to save
	def export_csv(self, in_path: str, in_df: pd.DataFrame) -> None:
		in_df.to_csv(in_path + "_" + "{:%Y_%m_%d_%H}".format(datetime.datetime.now()) + ".csv", index = False)


#---------------------------------------------- Dictionaries ----------------------------------------------
# Dictionaries with the queries and the process they need to be fetched
# Dictionary structure =>
# dictionary = ["SQL query": string, ["Titles from the set and the subsets fetched from the database"]: list,
# 				["Functions that require the subset to be merged"]: list]
dict_retired = {
#***************************************************************************************************
"query_1": ["\
SELECT P.CONSECUTIVO AS PERSONA, CASE WHEN  H.ASIGNACIONES_FACTURADAS IS NULL THEN 0 ELSE H.ASIGNACIONES_FACTURADAS END AS ASIGNACIONES_FACTURADAS \
FROM PERSONAS P \
LEFT JOIN ( \
    SELECT P.CONSECUTIVO AS PERSONA, COUNT(A.TARIFA_HORA) AS ASIGNACIONES_FACTURADAS \
    FROM ASIGNACIONES A \
    INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO = AH.CON_ASIG \
    INNER JOIN PERSONAS P                 ON P.CONSECUTIVO = A.CON_PERSONA \
    WHERE AH.FECHA_FIN > ADD_MONTHS(P.FECHA_RETIRO, -14) \
    AND   AH.FECHA_FIN < ADD_MONTHS(P.FECHA_RETIRO, -2) \
    AND   A.TARIFA_HORA > 0 \
    GROUP BY P.CONSECUTIVO) H ON P.CONSECUTIVO = H.PERSONA \
WHERE P.FECHA_RETIRO IS NOT NULL \
AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
ORDER BY P.CONSECUTIVO", ["ASIGNACIONES_FACTURADAS"], ["no_function"]],

#***************************************************************************************************
"query_2": ["\
SELECT P.CONSECUTIVO AS PERSONA, CASE WHEN  H.ASIGNACIONES_NO_FACTURADAS IS NULL THEN 0 ELSE H.ASIGNACIONES_NO_FACTURADAS END AS ASIGNACIONES_NO_FACTURADAS \
FROM PERSONAS P \
LEFT JOIN ( \
    SELECT P.CONSECUTIVO AS PERSONA, COUNT(A.TARIFA_HORA) AS ASIGNACIONES_NO_FACTURADAS \
    FROM ASIGNACIONES A \
    INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO = AH.CON_ASIG \
    INNER JOIN PERSONAS P                 ON P.CONSECUTIVO = A.CON_PERSONA \
    WHERE AH.FECHA_FIN > ADD_MONTHS(P.FECHA_RETIRO, -14) \
    AND   AH.FECHA_FIN < ADD_MONTHS(P.FECHA_RETIRO, -2) \
    AND   A.TARIFA_HORA = 0 \
    GROUP BY P.CONSECUTIVO) H ON P.CONSECUTIVO = H.PERSONA \
WHERE P.FECHA_RETIRO IS NOT NULL \
AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3  \
AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
ORDER BY P.CONSECUTIVO", ["ASIGNACIONES_NO_FACTURADAS"], ["no_function"]],

#***************************************************************************************************
"query_3": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.CANTIDAD_TIPOS_SERVICIO IS NULL THEN 0 ELSE B.CANTIDAD_TIPOS_SERVICIO END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
            FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND   P.FECHA_RETIRO IS NOT NULL \
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.TIPO AS TIPO_SERVICIO, COUNT(S.TIPO) AS CANTIDAD_TIPOS_SERVICIO \
	FROM ASIGNACIONES A \
	INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO  = AH.CON_ASIG \
	INNER JOIN PERSONAS P                 ON A.CON_PERSONA  = P.CONSECUTIVO \
	INNER JOIN SERVICIOS S                ON A.COD_SERVICIO = S.CODIGO_SERVICIO \
	WHERE AH.FECHA_FIN > ADD_MONTHS(P.FECHA_RETIRO,-14) \
	AND   AH.FECHA_FIN < ADD_MONTHS(P.FECHA_RETIRO ,-2) \
	AND   P.FECHA_RETIRO IS NOT NULL \
	GROUP BY P.CONSECUTIVO, S.TIPO ) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA,A.TIPO_SERVICIO", ["CANTIDAD_TIPOS_SERVICIO", "TIPO_SERVICIO"], ["pivot_table", "suffix", "TP"]],

#***************************************************************************************************
"query_4": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.HORAS_TIPO_SERVICIO IS NULL THEN 0 ELSE B.HORAS_TIPO_SERVICIO END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND   P.FECHA_RETIRO IS NOT NULL \
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT \
	P.CONSECUTIVO          AS PERSONA, \
	S.TIPO                 AS TIPO_SERVICIO, \
	SUM(EA.DURACION_HORAS) AS HORAS_TIPO_SERVICIO \
	FROM EJECUCION_ACTIVIDADES EA \
	INNER JOIN ENTREGABLES_SERVICIO ES ON EA.CON_ENTREGABLE_SERV = ES.CONSECUTIVO \
	INNER JOIN SERVICIOS S             ON ES.COD_SERVICIO        = S.CODIGO_SERVICIO \
	INNER JOIN PERSONAS  P             ON EA.CONSECUTIVO_PERSONA = P.CONSECUTIVO \
	WHERE EA.HORA_FINALIZACION > ADD_MONTHS(P.FECHA_RETIRO, -14) \
	AND   EA.HORA_FINALIZACION < ADD_MONTHS(P.FECHA_RETIRO , -2) \
	GROUP BY P.CONSECUTIVO, S.TIPO ) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA,A.TIPO_SERVICIO", ["HORAS_TIPO_SERVICIO", "TIPO_SERVICIO"], ["pivot_table", "suffix", "HR"]],

#***************************************************************************************************
"query_5": ["\
SELECT A.PERSONA, A.HERRAMIENTA, CASE WHEN B.VECES_HERRAMIENTA_ASIGNADA IS NULL THEN 0 ELSE B.VECES_HERRAMIENTA_ASIGNADA END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA,H.NOMBRE AS HERRAMIENTA \
	FROM HERRAMIENTAS H \
	CROSS JOIN PERSONAS P \
    WHERE P.FECHA_RETIRO IS NOT NULL\
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,H.NOMBRE) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO  AS PERSONA,H.NOMBRE  AS HERRAMIENTA,COUNT(AH.CONSECUTIVO) AS VECES_HERRAMIENTA_ASIGNADA \
	FROM HERRAMIENTAS_UTILIZADAS HU \
	INNER JOIN HERRAMIENTAS H             ON HU.CON_HERRAMIENTAS = H.CONSECUTIVO \
	INNER JOIN SERVICIOS S                ON HU.COD_SERVICIO     = S.CODIGO_SERVICIO \
	INNER JOIN ASIGNACIONES A             ON S.CODIGO_SERVICIO   = A.COD_SERVICIO \
	INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO       = AH.CON_ASIG \
	INNER JOIN PERSONAS P                 ON P.CONSECUTIVO       = A.CON_PERSONA \
	WHERE AH.FECHA_FIN > ADD_MONTHS(P.FECHA_RETIRO, -14) \
	AND   AH.FECHA_FIN < ADD_MONTHS(P.FECHA_RETIRO, -2) \
	AND   P.FECHA_RETIRO IS NOT NULL \
	GROUP BY P.CONSECUTIVO, H.NOMBRE) B ON A.PERSONA = B.PERSONA AND  A.HERRAMIENTA = B.HERRAMIENTA \
ORDER BY A.PERSONA, A.HERRAMIENTA", ["VECES_HERRAMIENTA_ASIGNADA", "HERRAMIENTA"], ["pivot_table"]],

#***************************************************************************************************
"query_6": ["\
SELECT A.PERSONA, A.ETAPA, CASE WHEN B.HORAS_ETAPA IS NULL THEN 0 ELSE B.HORAS_ETAPA END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, E.CODIGO_ETAPA  AS ETAPA \
	FROM ETAPAS  E \
	CROSS JOIN PERSONAS P \
    WHERE P.FECHA_RETIRO IS NOT NULL\
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,E.CODIGO_ETAPA) A \
LEFT JOIN ( \
	SELECT \
	P.CONSECUTIVO          AS PERSONA, \
	EA.CODIGO_ETAPA        AS ETAPA, \
	SUM(EA.DURACION_HORAS) AS HORAS_ETAPA \
	FROM EJECUCION_ACTIVIDADES EA \
	INNER JOIN PERSONAS P ON EA.CONSECUTIVO_PERSONA = P.CONSECUTIVO \
	WHERE EA.HORA_FINALIZACION > ADD_MONTHS(P.FECHA_RETIRO, -14) \
	AND   EA.HORA_FINALIZACION < ADD_MONTHS(P.FECHA_RETIRO, -2) \
	GROUP BY P.CONSECUTIVO, EA.CODIGO_ETAPA ) B ON A.PERSONA = B.PERSONA AND A.ETAPA = B.ETAPA \
ORDER BY A.PERSONA, A.ETAPA", ["HORAS_ETAPA", "ETAPA"], ["pivot_table", "fillna", "SIN_ETAPA"]],

#***************************************************************************************************
"query_7": ["\
SELECT B.PERSONA, B.AREA, CASE WHEN C.HORAS_AREA IS NULL THEN 0 ELSE C.HORAS_AREA END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, A.CONSECUTIVO  AS AREA \
	FROM AREAS  A \
	CROSS JOIN PERSONAS P \
    WHERE P.FECHA_RETIRO IS NOT NULL \
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,A.CONSECUTIVO  ) B \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.CON_AREA  AS AREA, SUM(EA.DURACION_HORAS) AS HORAS_AREA \
	FROM EJECUCION_ACTIVIDADES EA \
	INNER JOIN ENTREGABLES_SERVICIO ES ON EA.CON_ENTREGABLE_SERV = ES.CONSECUTIVO \
	INNER JOIN PERSONAS P              ON EA.CONSECUTIVO_PERSONA = P.CONSECUTIVO \
	INNER JOIN SERVICIOS S             ON ES.COD_SERVICIO  = S.CODIGO_SERVICIO \
	WHERE EA.HORA_FINALIZACION > ADD_MONTHS(P.FECHA_RETIRO, -14) \
	AND   EA.HORA_FINALIZACION < ADD_MONTHS(P.FECHA_RETIRO, -2) \
	GROUP BY P.CONSECUTIVO, S.CON_AREA ) C ON B.PERSONA = C.PERSONA AND C.PERSONA =  B.AREA \
ORDER BY B.PERSONA, B.AREA", ["HORAS_AREA", "AREA"], ["pivot_table"]],

#***************************************************************************************************
"query_8": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.CANTIDAD_PENDIENTES IS NULL THEN 0 ELSE B.CANTIDAD_PENDIENTES END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND   P.FECHA_RETIRO IS NOT NULL \
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.TIPO AS TIPO_SERVICIO, COUNT(PE.CODIGO_PENDIENTE) AS CANTIDAD_PENDIENTES \
	FROM ASIGNACIONES_HISTORICOS AH \
	INNER JOIN ASIGNACIONES A  ON A.CONSECUTIVO = AH.CON_ASIG \
	INNER JOIN PERSONAS     P  ON A.CON_PERSONA = P.CONSECUTIVO \
	INNER JOIN PENDIENTES   PE ON A.COD_SERVICIO = PE.CODIGO_SERVICIO \
	INNER JOIN SERVICIOS    S  ON PE.CODIGO_SERVICIO = S.CODIGO_SERVICIO \
	WHERE AH.FECHA_FIN > ADD_MONTHS(P.FECHA_RETIRO, -14) \
	AND   AH.FECHA_FIN < ADD_MONTHS(P.FECHA_RETIRO, -2) \
	AND   PE.FECHA_DETECCION > ADD_MONTHS(P.FECHA_RETIRO, -14) \
	AND   PE.FECHA_DETECCION < ADD_MONTHS(P.FECHA_RETIRO, -2) \
	GROUP BY P.CONSECUTIVO, S.TIPO) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA,A.TIPO_SERVICIO", ["CANTIDAD_PENDIENTES", "TIPO_SERVICIO"], ["pivot_table", "suffix", "PN"]],

#***************************************************************************************************
"query_9": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.CANTIDAD_RIESGOS IS NULL THEN 0 ELSE B.CANTIDAD_RIESGOS END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND   P.FECHA_RETIRO IS NOT NULL \
    AND  MONTHS_BETWEEN (P.FECHA_RETIRO, P.FECHA_INGRESO) >= 3 \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.TIPO AS TIPO_SERVICIO, COUNT(RS.CONSECUTIVO) AS CANTIDAD_RIESGOS \
	FROM ASIGNACIONES_HISTORICOS AH \
	INNER JOIN ASIGNACIONES A      ON A.CONSECUTIVO = AH.CON_ASIG \
	INNER JOIN PERSONAS     P      ON A.CON_PERSONA = P.CONSECUTIVO \
	INNER JOIN RIESGOS_SERVICIO RS ON A.COD_SERVICIO = RS.CODIGO_SERVICIO \
	INNER JOIN SERVICIOS    S      ON RS.CODIGO_SERVICIO = S.CODIGO_SERVICIO \
	WHERE AH.FECHA_FIN BETWEEN ADD_MONTHS(P.FECHA_RETIRO, -14) AND ADD_MONTHS(P.FECHA_RETIRO, -2) \
	AND   RS.FECHA_REGISTRO BETWEEN ADD_MONTHS(P.FECHA_RETIRO, -14)  AND ADD_MONTHS(P.FECHA_RETIRO, -2) \
	GROUP BY P.CONSECUTIVO, S.TIPO ) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA, A.TIPO_SERVICIO", ["CANTIDAD_RIESGOS", "TIPO_SERVICIO"], ["pivot_table", "suffix", "RS"]],

#***************************************************************************************************
"query_10": ["SELECT CONSECUTIVO AS PERSONA, FECHA_NACIMIENTO, FECHA_RETIRO FROM PERSONAS\
            WHERE FECHA_RETIRO IS NOT NULL AND FECHA_NACIMIENTO IS NOT NULL", 
			["EDAD", "FECHA_NACIMIENTO", "FECHA_RETIRO"], ["age", "apply", "drop"]],

#***************************************************************************************************
"query_11": ["SELECT CONSECUTIVO AS PERSONA, GENERO FROM PERSONAS\
            WHERE FECHA_RETIRO IS NOT NULL AND FECHA_NACIMIENTO IS NOT NULL", ["GENERO"], ["replace"]],

#***************************************************************************************************
"query_12": ["SELECT CONSECUTIVO AS PERSONA, FECHA_INGRESO, FECHA_RETIRO FROM PERSONAS\
            WHERE FECHA_RETIRO IS NOT NULL AND FECHA_INGRESO IS NOT NULL", 
			["TIEMPO_EMP", "FECHA_INGRESO", "FECHA_RETIRO"], ["time", "apply", "drop"]]

#***************************************************************************************************
}

dict_non_retired = {
#***************************************************************************************************
"query_1": ["\
SELECT P.CONSECUTIVO AS PERSONA, CASE WHEN  H.ASIGNACIONES_FACTURADAS IS NULL THEN 0 ELSE H.ASIGNACIONES_FACTURADAS END AS ASIGNACIONES_FACTURADAS \
FROM PERSONAS P \
LEFT JOIN ( \
SELECT P.CONSECUTIVO AS PERSONA, COUNT(A.TARIFA_HORA) AS ASIGNACIONES_FACTURADAS \
FROM ASIGNACIONES A \
INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO = AH.CON_ASIG \
INNER JOIN PERSONAS P                 ON P.CONSECUTIVO = A.CON_PERSONA \
WHERE AH.FECHA_FIN > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
AND   AH.FECHA_FIN < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
AND   A.TARIFA_HORA > 0 \
GROUP BY P.CONSECUTIVO) H ON P.CONSECUTIVO = H.PERSONA \
WHERE P.FECHA_RETIRO IS NULL \
AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
ORDER BY P.CONSECUTIVO", ["ASIGNACIONES_FACTURADAS"], ["no_function"]],

#***************************************************************************************************
"query_2": ["\
SELECT P.CONSECUTIVO AS PERSONA, CASE WHEN  H.ASIGNACIONES_NO_FACTURADAS IS NULL THEN 0 ELSE H.ASIGNACIONES_NO_FACTURADAS END AS ASIGNACIONES_NO_FACTURADAS \
FROM PERSONAS P \
LEFT JOIN ( \
SELECT P.CONSECUTIVO AS PERSONA, COUNT(A.TARIFA_HORA) AS ASIGNACIONES_NO_FACTURADAS \
FROM ASIGNACIONES A \
INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO = AH.CON_ASIG \
INNER JOIN PERSONAS P                 ON P.CONSECUTIVO = A.CON_PERSONA \
WHERE AH.FECHA_FIN > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
AND   AH.FECHA_FIN < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
AND   A.TARIFA_HORA = 0 \
GROUP BY P.CONSECUTIVO) H ON P.CONSECUTIVO = H.PERSONA \
WHERE P.FECHA_RETIRO IS NULL \
AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
AND   P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
ORDER BY P.CONSECUTIVO", ["ASIGNACIONES_NO_FACTURADAS"], ["no_function"]],

#***************************************************************************************************
"query_3": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.CANTIDAD_TIPOS_SERVICIO IS NULL THEN 0 ELSE B.CANTIDAD_TIPOS_SERVICIO END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
	AND C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND P.FECHA_RETIRO IS NULL \
    AND P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.TIPO AS TIPO_SERVICIO, COUNT(S.TIPO) AS CANTIDAD_TIPOS_SERVICIO \
	FROM ASIGNACIONES A \
	INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO  = AH.CON_ASIG \
	INNER JOIN PERSONAS P                 ON A.CON_PERSONA  = P.CONSECUTIVO \
	INNER JOIN SERVICIOS S                ON A.COD_SERVICIO = S.CODIGO_SERVICIO \
	WHERE P.ESTADO = 1 \
	AND AH.FECHA_FIN > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-14) \
	AND   AH.FECHA_FIN < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-2) \
	AND   P.FECHA_RETIRO IS NULL \
	GROUP BY P.CONSECUTIVO, S.TIPO ) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA,A.TIPO_SERVICIO", ["CANTIDAD_TIPOS_SERVICIO", "TIPO_SERVICIO"], ["pivot_table", "suffix", "TP"]],

#***************************************************************************************************
"query_4": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.HORAS_TIPO_SERVICIO IS NULL THEN 0 ELSE B.HORAS_TIPO_SERVICIO END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
	AND C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND P.FECHA_RETIRO IS NULL \
    AND P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT \
	P.CONSECUTIVO          AS PERSONA, \
	S.TIPO                 AS TIPO_SERVICIO, \
	SUM(EA.DURACION_HORAS) AS HORAS_TIPO_SERVICIO \
	FROM EJECUCION_ACTIVIDADES EA \
	INNER JOIN ENTREGABLES_SERVICIO ES ON EA.CON_ENTREGABLE_SERV = ES.CONSECUTIVO \
	INNER JOIN SERVICIOS S             ON ES.COD_SERVICIO        = S.CODIGO_SERVICIO \
	INNER JOIN PERSONAS  P             ON EA.CONSECUTIVO_PERSONA = P.CONSECUTIVO \
	WHERE P.ESTADO = 1 \
	AND EA.HORA_FINALIZACION > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
	AND   EA.HORA_FINALIZACION < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE) , -2) \
	GROUP BY P.CONSECUTIVO, S.TIPO ) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA,A.TIPO_SERVICIO", ["HORAS_TIPO_SERVICIO", "TIPO_SERVICIO"], ["pivot_table", "suffix", "HR"]],

#***************************************************************************************************
"query_5": ["\
SELECT A.PERSONA, A.HERRAMIENTA, CASE WHEN B.VECES_HERRAMIENTA_ASIGNADA IS NULL THEN 0 ELSE B.VECES_HERRAMIENTA_ASIGNADA END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA,H.NOMBRE AS HERRAMIENTA \
	FROM HERRAMIENTAS H	 \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
	AND   P.FECHA_RETIRO IS NULL \
    AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,H.NOMBRE) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO  AS PERSONA,H.NOMBRE  AS HERRAMIENTA,COUNT(AH.CONSECUTIVO) AS VECES_HERRAMIENTA_ASIGNADA \
	FROM HERRAMIENTAS_UTILIZADAS HU \
	INNER JOIN HERRAMIENTAS H             ON HU.CON_HERRAMIENTAS = H.CONSECUTIVO \
	INNER JOIN SERVICIOS S                ON HU.COD_SERVICIO     = S.CODIGO_SERVICIO \
	INNER JOIN ASIGNACIONES A             ON S.CODIGO_SERVICIO   = A.COD_SERVICIO \
	INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO       = AH.CON_ASIG \
	INNER JOIN PERSONAS P                 ON P.CONSECUTIVO       = A.CON_PERSONA \
	WHERE P.ESTADO = 1 \
	AND   AH.FECHA_FIN > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
	AND   AH.FECHA_FIN < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	AND   P.FECHA_RETIRO IS NULL \
	GROUP BY P.CONSECUTIVO, H.NOMBRE) B ON A.PERSONA = B.PERSONA AND  A.HERRAMIENTA = B.HERRAMIENTA \
ORDER BY A.PERSONA, A.HERRAMIENTA", ["VECES_HERRAMIENTA_ASIGNADA", "HERRAMIENTA"], ["pivot_table"]],

#***************************************************************************************************
"query_6": ["\
SELECT A.PERSONA, A.ETAPA, CASE WHEN B.HORAS_ETAPA IS NULL THEN 0 ELSE B.HORAS_ETAPA END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, E.CODIGO_ETAPA  AS ETAPA \
	FROM ETAPAS  E \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
    AND   P.FECHA_RETIRO IS NULL \
    AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO, E.CODIGO_ETAPA) A \
LEFT JOIN ( \
	SELECT \
	P.CONSECUTIVO          AS PERSONA, \
	EA.CODIGO_ETAPA        AS ETAPA, \
	SUM(EA.DURACION_HORAS) AS HORAS_ETAPA \
	FROM EJECUCION_ACTIVIDADES EA \
	INNER JOIN PERSONAS P ON EA.CONSECUTIVO_PERSONA = P.CONSECUTIVO \
	WHERE P.ESTADO = 1 \
	AND   EA.HORA_FINALIZACION > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
	AND   EA.HORA_FINALIZACION < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	GROUP BY P.CONSECUTIVO, EA.CODIGO_ETAPA ) B ON A.PERSONA = B.PERSONA AND A.ETAPA = B.ETAPA \
ORDER BY A.PERSONA, A.ETAPA", ["HORAS_ETAPA", "ETAPA"], ["pivot_table", "fillna", "SIN_ETAPA"]],

#***************************************************************************************************
"query_7": ["\
SELECT B.PERSONA, B.AREA, CASE WHEN C.HORAS_AREA IS NULL THEN 0 ELSE C.HORAS_AREA END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, A.CONSECUTIVO  AS AREA \
	FROM AREAS  A \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
    AND   P.FECHA_RETIRO IS NULL \
    AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND   P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,A.CONSECUTIVO  ) B \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.CON_AREA  AS AREA, SUM(EA.DURACION_HORAS) AS HORAS_AREA \
	FROM EJECUCION_ACTIVIDADES EA \
	INNER JOIN ENTREGABLES_SERVICIO ES ON EA.CON_ENTREGABLE_SERV = ES.CONSECUTIVO \
	INNER JOIN PERSONAS P              ON EA.CONSECUTIVO_PERSONA = P.CONSECUTIVO \
	INNER JOIN SERVICIOS S             ON ES.COD_SERVICIO  = S.CODIGO_SERVICIO \
	WHERE P.ESTADO = 1 \
	AND   EA.HORA_FINALIZACION > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
	AND   EA.HORA_FINALIZACION < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	GROUP BY P.CONSECUTIVO, S.CON_AREA ) C ON B.PERSONA = C.PERSONA AND C.PERSONA =  B.AREA \
ORDER BY B.PERSONA, B.AREA", ["HORAS_AREA", "AREA"], ["pivot_table"]],

#***************************************************************************************************
"query_8": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.CANTIDAD_PENDIENTES IS NULL THEN 0 ELSE B.CANTIDAD_PENDIENTES END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
	AND   C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND   P.FECHA_RETIRO IS NULL \
    AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND   P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.TIPO AS TIPO_SERVICIO, COUNT(PE.CODIGO_PENDIENTE) AS CANTIDAD_PENDIENTES \
	FROM ASIGNACIONES_HISTORICOS AH \
	INNER JOIN ASIGNACIONES A  ON A.CONSECUTIVO = AH.CON_ASIG \
	INNER JOIN PERSONAS     P  ON A.CON_PERSONA = P.CONSECUTIVO \
	INNER JOIN PENDIENTES   PE ON A.COD_SERVICIO = PE.CODIGO_SERVICIO \
	INNER JOIN SERVICIOS    S  ON PE.CODIGO_SERVICIO = S.CODIGO_SERVICIO \
	WHERE P.ESTADO = 1 \
	AND   AH.FECHA_FIN > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
	AND   AH.FECHA_FIN < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	AND   AH.FECHA_FIN >= AH.FECHA_INICIO \
	AND   PE.FECHA_DETECCION > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) \
	AND   PE.FECHA_DETECCION < ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	GROUP BY P.CONSECUTIVO, S.TIPO) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA,A.TIPO_SERVICIO", ["CANTIDAD_PENDIENTES", "TIPO_SERVICIO"], ["pivot_table", "suffix", "PN"]],

#***************************************************************************************************
"query_9": ["\
SELECT A.PERSONA, A.TIPO_SERVICIO, CASE WHEN B.CANTIDAD_RIESGOS IS NULL THEN 0 ELSE B.CANTIDAD_RIESGOS END \
FROM ( \
	SELECT P.CONSECUTIVO AS PERSONA, C.RV_LOW_VALUE AS TIPO_SERVICIO \
	FROM CG_REF_CODES C \
	CROSS JOIN PERSONAS P \
	WHERE P.ESTADO = 1 \
	AND   C.RV_DOMAIN = 'TIPO_PROYECTO' \
	AND   P.FECHA_RETIRO IS NULL \
    AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-5) \
    AND   P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
	GROUP BY P.CONSECUTIVO,C.RV_LOW_VALUE ) A \
LEFT JOIN ( \
	SELECT P.CONSECUTIVO AS PERSONA, S.TIPO AS TIPO_SERVICIO, COUNT(RS.CONSECUTIVO) AS CANTIDAD_RIESGOS \
	FROM ASIGNACIONES_HISTORICOS AH \
	INNER JOIN ASIGNACIONES A      ON A.CONSECUTIVO = AH.CON_ASIG \
	INNER JOIN PERSONAS     P      ON A.CON_PERSONA = P.CONSECUTIVO \
	INNER JOIN RIESGOS_SERVICIO RS ON A.COD_SERVICIO = RS.CODIGO_SERVICIO \
	INNER JOIN SERVICIOS    S      ON RS.CODIGO_SERVICIO = S.CODIGO_SERVICIO \
	WHERE P.ESTADO = 1 \
	AND   AH.FECHA_FIN BETWEEN ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14) AND ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	AND   AH.FECHA_FIN > AH.FECHA_INICIO \
	AND   RS.FECHA_REGISTRO BETWEEN ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -14)  AND ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -2) \
	GROUP BY P.CONSECUTIVO, S.TIPO ) B ON A.PERSONA = B.PERSONA AND A.TIPO_SERVICIO = B.TIPO_SERVICIO \
ORDER BY A.PERSONA, A.TIPO_SERVICIO", ["CANTIDAD_RIESGOS", "TIPO_SERVICIO"], ["pivot_table", "suffix", "RS"]],

#***************************************************************************************************
"query_10": ["SELECT CONSECUTIVO AS PERSONA, FECHA_NACIMIENTO FROM PERSONAS\
            WHERE FECHA_RETIRO IS NULL AND FECHA_NACIMIENTO IS NOT NULL", 
			["EDAD", "FECHA_NACIMIENTO"], ["age", "apply", "drop"]],

#***************************************************************************************************
"query_11": ["SELECT CONSECUTIVO AS PERSONA, GENERO FROM PERSONAS\
            WHERE FECHA_RETIRO IS NULL AND FECHA_NACIMIENTO IS NOT NULL", ["GENERO"], ["replace"]],

#***************************************************************************************************
"query_12": ["SELECT CONSECUTIVO AS PERSONA, FECHA_INGRESO FROM PERSONAS\
            WHERE FECHA_RETIRO IS NULL AND FECHA_INGRESO IS NOT NULL", 
			["TIEMPO_EMP", "FECHA_INGRESO"], ["time", "apply", "drop"]]

#***************************************************************************************************
}

#---------------------------------------------- Calls ----------------------------------------------
# Reading querys of retired people
f_retired = open("../static/data/query_retired.json")
querys_retired = json.load(f_retired)

f_non_retired = open("../static/data/query_non_retired.json")
querys_non_retired = json.load(f_non_retired)

con = Connection()
con.try_connection()
dic_df_retired = con.make_query(querys_retired)

print(dic_df_retired)

dic_df_non_retired = con.make_query(querys_non_retired)

df_retired = con.merge_df(dic_df_retired)
df_retired.head()
print(list(df_retired.columns))

df_non_retired = con.merge_df(dic_df_non_retired)
df_non_retired.head()

print(list(df_non_retired.columns))

con.export_csv(in_path = "../static/data/raw_data/retired", in_df = df_retired)
con.export_csv(in_path = "../static/data/raw_data/non_retired", in_df = df_non_retired)

#https://loaddatafunc.blob.core.windows.net/churn-files











