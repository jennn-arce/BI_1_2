from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Any
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from support_functions import loadModel, createModel, transfor_data_local

# Make sure transfor_data_local is accessible globally
globals()['transfor_data_local'] = transfor_data_local

# Initialize FastAPI application
app = FastAPI(
    title="Predictor API",
    description="API para predicciones y re-entrenamiento de modelo."
)


class Instance(BaseModel):
    Titulo: str = Field(..., example="Título de ejemplo")
    Descripcion: str = Field(..., example="Descripción de ejemplo")
    ID: Optional[str] = Field(None, example="123")
    Fecha: Optional[str] = Field(None, example="2025-03-25")


class PredictionResult(BaseModel):
    prediction: Any
    # probability may be None if model does not support it
    probability: Optional[float]


class RetrainInstance(Instance):
    # Suponiendo una variable binaria; ajustar según sea necesario.
    Label: int = Field(..., example=1)


MODEL_PATH = "model/Predictor.joblib"
model = loadModel(MODEL_PATH)


@app.post("/predict", response_model=List[PredictionResult])
def predict(request: List[Instance]):
    """
    Endpoint para realizar predicciones.
    Recibe una o más instancias de datos (con 'Titulo' y 'Descripcion') y devuelve la predicción y probabilidad para cada instancia.
    """
    try:
        # Convertir las instancias a un DataFrame usando solo 'Titulo' y 'Descripcion'
        df = pd.DataFrame([{"Titulo": instance.Titulo, "Descripcion": instance.Descripcion}
                           for instance in request])
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error procesando los datos de entrada: {e}"
        )

    try:
        preds = model.predict(df)
        # Si el modelo soporta predict_proba, se obtiene la probabilidad de la clase predicha.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            # Para cada instancia, obtener la probabilidad asociada a la clase predicha.
            pred_probs = [p[1] if pred == 1 else p[0]
                          for pred, p in zip(preds, proba)]
        else:
            pred_probs = [None] * len(preds)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error durante la predicción: {e}"
        )

    return [PredictionResult(prediction=int(pred), probability=prob)
            for pred, prob in zip(preds, pred_probs)]


@app.post("/retrain")
def retrain(request: List[RetrainInstance]):
    """
    Endpoint para re-entrenar el modelo.
    Recibe una lista de instancias que incluyen 'Titulo', 'Descripcion' y 'Label'.
    """
    try:
        # Crear DataFrame para las características: usamos solo 'Titulo' y 'Descripcion'
        df = pd.DataFrame([{"Titulo": instance.Titulo, "Descripcion": instance.Descripcion, "Label": instance.Label}
                          for instance in request])
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error procesando los datos de entrada: {e}"
        )

    try:
        text_transformer_basic = ColumnTransformer(
            transformers=[
                ("Titulo_tfidf", TfidfVectorizer(), "Titulo"),
                ("desc_tfidf", TfidfVectorizer(), "Descripcion")
            ]
        )
        results, new_model = createModel(text_transformer_basic, df)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error durante el re-entrenamiento: {e}"
        )

    try:
        # Guardar el nuevo modelo reemplazando el archivo existente.
        joblib.dump(new_model, MODEL_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error guardando el nuevo modelo: {e}"
        )

    # Actualizar la variable global del modelo para futuras predicciones.
    global model
    model = new_model

    return {
        "message": "Modelo re-entrenado y actualizado correctamente.",
        "metrics": results
    }


# Para ejecutar la API, utiliza el comando: uvicorn main:app --reload
if __name__ == "__main__":
    import hypercorn.asyncio
    import asyncio

    async def serve_app():
        config = hypercorn.Config()
        config.bind = ["0.0.0.0:8000"]
        await hypercorn.asyncio.serve(app, config)

    asyncio.run(serve_app())
