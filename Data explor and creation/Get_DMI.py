# Databricks notebook source
from pyspark.sql import SparkSession
from requests.auth import HTTPBasicAuth
import requests
import urllib.parse

# COMMAND ----------

# import josn

url = 'https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items'
api_key = '49707636-6629-4fd1-bbd0-047824c3d013'
#auth = HTTPBasicAuth('api-key', api_key)

# 2020-01-01 00:00:00+00:00
#2024-02-22 11:00:00+00:00

date = '2020-01-01T00:00:00.000Z/2024-02-22T00:00:00.000Z'

params = {
        'api-key': api_key,
        'municipalityId' : '0101', 
        'parameterId':'mean_temp', 
        'timeResolution':'hour', 
        'datetime':date
        }

req = requests.get(url, params=params)

data = req.json()

data_list = [data]

numberReturned = data['numberReturned']


i = 0
while numberReturned != 0:
    req = requests.get(data['links'][1]['href'])
    data = req.json()
    data_list += [data]
    numberReturned = data['numberReturned']
    i += 1
    print('iteratin: ', i)

data_list.pop(-1)

#print(data_list)

print(len(data_list))

# df = spark.createDataFrame(data_list['features'])
# display(df)



# COMMAND ----------

# MAGIC %md
# MAGIC https://dmigw.govcloud.dk/v2/climateData/collections/municipalityValue/items?api-key=49707636-6629-4fd1-bbd0-047824c3d013&municipalityId=0101&parameterId=mean_temp&timeResolution=hour&datetime=2023-01-15T23:00:00.000Z/2023-01-17T22:00:00.000Z

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# display(df.select('*','properties.value'))
# string = data_list[0]['features'][0]['properties'].keys()
# print(string)
# display(df.select('id',*[f'properties.{k}' for k in string]))


# COMMAND ----------

from itertools import chain

# flatten list
flat_data_list = [c for l in data_list for c in l['features']]

flat_spark = spark.createDataFrame(flat_data_list)

relevant_features = ['from', 'to', 'municipalityName', 'parameterId', 'value']

new_df = flat_spark.select('id',*[f'properties.{k}' for k in relevant_features])

print(new_df.count())

# COMMAND ----------

display(new_df)

# COMMAND ----------

# # Write DataFrame to Parquet table
# table_name = 'dmi'
# # delta_path = f"abfss://ML_Stab_TI_Energinet_EBI@onelake.dfs.fabric.microsoft.com/Ines.Lakehouse/Tables/{table_name}"

# # df.write.format("delta").save(delta_path)

# new_df.write.format("delta").saveAsTable(table_name)
