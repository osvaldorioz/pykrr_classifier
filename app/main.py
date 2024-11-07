from fastapi import FastAPI
from krr_classifier_mod import KRRClassifier
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class Vector(BaseModel):
    vector: List[float]

@app.post("/krr")
async def classifier(lambda_: float, 
                 X_train: Matrix,
                 y_train: Vector,
                 X_test: Matrix):
    start = time.time()
    
    krr = KRRClassifier(lambda_)
    krr.fit(X_train.matrix, y_train.vector)

    # Hacer predicciones
    #X_test = [[1.5, 2.5], [3.0, 5.0]]
    predictions = krr.predict(X_test.matrix)

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Predicciones": predictions
    }
    jj = json.dumps(j1)

    return jj