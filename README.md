# BI_1_2

## To run back-end:

### Pip install all the dependencies

pip install -r requirements.txt

cd back

uvicorn backend:app --host 0.0.0.0 --port 8000


## To run front-end:

cd front

npm install

npm run dev

## Archivos prueba

En la carpeta principal se encuentran 

- predict_data.json (para predecirlos y ver resultados)

- data_10%.json (para entrenar el modelo, tiene el 10% de los datos originales)

- data_20%.json (para entrenar el modelo, tiene el 20% de los datos originales)

- data_50%.json (para entrenar el modelo, tiene el 50% de los datos originales)

