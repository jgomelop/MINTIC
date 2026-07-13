# -*- coding: utf-8 -*-
"""Diagnóstico: carga el modelo y predice UNA fila, sin Streamlit de por
medio. Si esto revienta igual, el problema es 100% del stack
numpy/scipy/scikit-learn/lightgbm dentro del contenedor -- no de la app.

Uso dentro del contenedor:
  podman run --rm --entrypoint python localhost/riesgo-academico:dev \
      /app/test_modelo.py
"""
import sys

import joblib
import numpy as np
import pandas as pd

print("1) Importando librerías... OK", flush=True)
print(f"   numpy {np.__version__}", flush=True)
import scipy
print(f"   scipy {scipy.__version__}", flush=True)
import sklearn
print(f"   scikit-learn {sklearn.__version__}", flush=True)
import lightgbm
print(f"   lightgbm {lightgbm.__version__}", flush=True)
print(f"   Python {sys.version}", flush=True)

print("2) Cargando modelo_riesgo.pkl...", flush=True)
pipe = joblib.load("outputs/modelo_riesgo.pkl")
print("   Modelo cargado OK", flush=True)

CAT_FEATURES = ["SEXO", "SEDE", "FACULTAD", "TIPO_ACEPTACION",
                "NATURALEZA_COLE", "NIVEL_PREGRADO"]
NUM_FEATURES = ["EDAD", "ESTRATO", "ESTRATO_FALTANTE", "ANTIGUEDAD_SEMESTRES",
                "CREDITOS_ULTIM_SEMEST_MATRIC", "PCT_RURAL_MUNI",
                "PROM_SABER_MUNI", "TASA_BENEF_MUNI", "VIVE_FUERA_ANTIOQUIA"]

print("3) Extrayendo categorías del OneHotEncoder...", flush=True)
prep = pipe.named_steps["prep"]
ohe = prep.named_transformers_["cat"]
categorias = dict(zip(CAT_FEATURES, ohe.categories_))
for col, cats in categorias.items():
    print(f"   {col}: {list(cats)[:3]}... ({len(cats)} categorías)", flush=True)

print("4) Construyendo fila de prueba...", flush=True)
fila = pd.DataFrame([{
    "SEXO": categorias["SEXO"][0],
    "SEDE": categorias["SEDE"][0],
    "FACULTAD": categorias["FACULTAD"][0],
    "TIPO_ACEPTACION": categorias["TIPO_ACEPTACION"][0],
    "NATURALEZA_COLE": categorias["NATURALEZA_COLE"][0],
    "NIVEL_PREGRADO": categorias["NIVEL_PREGRADO"][0],
    "EDAD": 20.0,
    "ESTRATO": 3.0,
    "ESTRATO_FALTANTE": 0,
    "ANTIGUEDAD_SEMESTRES": 2,
    "CREDITOS_ULTIM_SEMEST_MATRIC": 16,
    "PCT_RURAL_MUNI": 30.0,
    "PROM_SABER_MUNI": 250.0,
    "TASA_BENEF_MUNI": 10.0,
    "VIVE_FUERA_ANTIOQUIA": 0,
}])[CAT_FEATURES + NUM_FEATURES]
print("   Fila lista", flush=True)

print("5) Llamando predict_proba...", flush=True)
prob = pipe.predict_proba(fila)[0, 1]
print(f"   OK -> probabilidad de riesgo: {prob:.4f}", flush=True)

print("TODO OK -- el stack de predicción funciona bien en este contenedor.")