import os
import pandas as pd
import datetime
from datetime import date
import cx_Oracle
import  config
import pandas as pd

q="\
SELECT P.CONSECUTIVO AS PERSONA, CASE WHEN  H.ASIGNACIONES_FACTURADAS IS NULL THEN 0 ELSE H.ASIGNACIONES_FACTURADAS END AS ASIGNACIONES_FACTURADAS \
FROM PERSONAS P \
LEFT JOIN ( \
SELECT P.CONSECUTIVO AS PERSONA, COUNT(A.TARIFA_HORA) AS ASIGNACIONES_FACTURADAS \
FROM ASIGNACIONES A \
INNER JOIN ASIGNACIONES_HISTORICOS AH ON A.CONSECUTIVO = AH.CON_ASIG \
INNER JOIN PERSONAS P                 ON P.CONSECUTIVO = A.CON_PERSONA \
WHERE AH.FECHA_FIN > ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE), -12) \
AND   AH.FECHA_FIN < CAST(CURRENT_TIMESTAMP AS DATE) \
AND   A.TARIFA_HORA > 0 \
GROUP BY P.CONSECUTIVO) H ON P.CONSECUTIVO = H.PERSONA \
WHERE P.FECHA_RETIRO IS NULL \
AND   P.FECHA_INGRESO <= ADD_MONTHS(CAST(CURRENT_TIMESTAMP AS DATE),-3) \
AND  P.GERENCIA_RESPONSABLE  <> 'INTELIGENCIA ARTIFICIAL'  \
ORDER BY P.CONSECUTIVO"

class conection():
    '''Conexión a la base de datos'''
    #cx_Oracle.init_oracle_client(lib_dir=config.lib_dir)
    def __init__(self):
        self.username= config.username
        self.password= config.password
        self.dsn = config.dsn
        self.encoding = config.encoding
        self.conection = None
        self.cur = None
    connection = None
    def try_conection(self):
        try:
            print("username:",self.username)
            self.conection = cx_Oracle.connect(
                self.username,
                self.password,
                self.dsn,
                encoding=self.encoding)
            # show the version of the Oracle Database
            print("conection_version:",self.conection.version)
        except cx_Oracle.Error as error:
            print("error:", error)
    def make_query (self,query):
        time_start = datetime.datetime.now()
        self.cur=self.conection.cursor()
        self.cur.execute(query)
        self.res=self.cur.fetchall()
        query_time = datetime.datetime.now() - time_start
        print(f"Duración de Consulta (seg): {query_time}")

con = conection()
con.try_conection()
con.make_query(q)


