from fastapi import FastAPI
from joblib import load
import uvicorn
import pandas as pd
from typing import List

app= FastAPI()

energy_consumption_model=load("C:\workspace\smart energy\saved project\energy_consumption.joblib")
anomalynergy_consumption_model=load(r"C:\workspace\smart energy\saved project\anomaly_energy_consumption_predict.joblib")
energy_leakage_model=load(r"C:\workspace\smart energy\saved project\energy_leakage.joblib")
loaded_objects = load('C:\workspace\smart energy\saved project\multiple_objects.joblib')

@app.get('/')
async def root():
    return '------ Application is Running! ------'

@app.post('/energy_consumption')
def predict(data: List[dict]):
    df=pd.DataFrame(data)
    df["building_category"]= loaded_objects[0]
    df["season"]= loaded_objects[1] 
    df["energy_consumption_predict"]=energy_consumption_model.predict(df)
    return df.to_dict(orient='records')

@app.post('/Anomaly_energy_consumption')
def predict(data: List[dict]):
    df=pd.DataFrame(data)
    df["building_category"]= loaded_objects[0]
    df["season"]= loaded_objects[1] 
    anomaly=anomalynergy_consumption_model.predict(df)
    df["energy_consumption_predict"]=["anomaly" if i==-1 else "not a anomaly" for i in anomaly]
    return df.to_dict(orient='records')

@app.post('/energy_leakageclassify')
def predict(data: List[dict]):
    df=pd.DataFrame(data)
    df["leakage"]=energy_leakage_model.predict(df[['bulk_consumption', '1_building', '2_building', '3_building',
       '4_building', '5_building', '6_building', '7_building']])
    return df.to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9777, log_level="info", reload=True)