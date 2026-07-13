# -*- coding: utf-8 -*-
"""App mínima: SOLO el formulario de predicción de riesgo académico.
Nada de pestañas, módulo A, SHAP ni tabla de estudiantes -- se usa para
aislar el problema de despliegue antes de reintroducir el resto.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import traceback

OUT = Path(__file__).resolve().parent / "outputs"

CAT_FEATURES = ["SEXO", "SEDE", "FACULTAD", "TIPO_ACEPTACION",
                "NATURALEZA_COLE", "NIVEL_PREGRADO"]
NUM_FEATURES = ["EDAD", "ESTRATO", "ESTRATO_FALTANTE", "ANTIGUEDAD_SEMESTRES",
                "CREDITOS_ULTIM_SEMEST_MATRIC", "PCT_RURAL_MUNI",
                "PROM_SABER_MUNI", "TASA_BENEF_MUNI", "VIVE_FUERA_ANTIOQUIA"]

st.set_page_config(page_title="Riesgo académico (mínimo)", page_icon="🎓")
print("A) set_page_config OK", flush=True)
st.title("Calculadora de riesgo académico")
print("B) st.title OK", flush=True)


@st.cache_resource
def cargar_modelo():
    print("C) entrando a cargar_modelo()", flush=True)
    m = joblib.load(OUT / "modelo_riesgo.pkl")
    print("D) joblib.load terminó OK", flush=True)
    return m


pipe = cargar_modelo()
print("E) cargar_modelo() retornó OK", flush=True)
st.success("Modelo cargado correctamente.")
print("F) st.success OK", flush=True)

prep = pipe.named_steps["prep"]
ohe = prep.named_transformers_["cat"]
categorias = {col: list(cats) for col, cats in zip(CAT_FEATURES, ohe.categories_)}
print("G) categorias extraídas OK", flush=True)

with st.form("form_prediccion"):
    print("H) entrando al with st.form", flush=True)
    c1, c2 = st.columns(2)
    print("I) columnas creadas OK", flush=True)
    with c1:
        sexo = st.selectbox("Sexo", categorias["SEXO"])
        print("I.1) sexo OK", flush=True)
        sede = st.selectbox("Sede", categorias["SEDE"])
        print("I.2) sede OK", flush=True)
        facultad = st.selectbox("Facultad", categorias["FACULTAD"])
        print("I.3) facultad OK", flush=True)
        tipo_acept = st.selectbox("Tipo de aceptación", categorias["TIPO_ACEPTACION"])
        print("I.4) tipo_acept OK", flush=True)
        nat_cole = st.selectbox("Naturaleza del colegio", categorias["NATURALEZA_COLE"])
        print("I.5) nat_cole OK", flush=True)
        nivel_pregrado = st.selectbox("Nivel de pregrado", categorias["NIVEL_PREGRADO"])
        print("I.6) nivel_pregrado OK", flush=True)
    with c2:
        edad = st.number_input("Edad", min_value=15, max_value=70, value=20)
        print("I.7) edad OK", flush=True)
        sin_estrato = st.checkbox("No conoce / no aplica estrato")
        print("I.8) sin_estrato OK", flush=True)
        estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6], disabled=sin_estrato)
        print("I.9) estrato OK", flush=True)
        antiguedad = st.number_input("Antigüedad (semestres)", min_value=1, max_value=40, value=1)
        print("I.10) antiguedad OK", flush=True)
        creditos_ult = st.number_input("Créditos último semestre", min_value=0, max_value=30, value=16)
        print("I.11) creditos_ult OK", flush=True)
        vive_fuera = st.checkbox("Vive fuera de Antioquia")
        print("I.12) vive_fuera OK", flush=True)

    pct_rural = st.number_input("% población rural del municipio", 0.0, 100.0, 30.0)
    print("I.13) pct_rural OK", flush=True)
    prom_saber = st.number_input("Promedio Saber 11 del municipio", 0.0, 500.0, 250.0)
    print("I.14) prom_saber OK", flush=True)
    tasa_benef = st.number_input("Tasa de acompañamiento del municipio (/1.000 jóv.)", 0.0, 500.0, 10.0)
    print("I.15) tasa_benef OK", flush=True)

    enviado = st.form_submit_button("Calcular riesgo")

print("J) script terminó de correr completo, esperando interacción", flush=True)

if enviado:
    print("K) formulario enviado, construyendo fila...", flush=True)
    try:
        datos = {
            "SEXO": sexo, "SEDE": sede, "FACULTAD": facultad,
            "TIPO_ACEPTACION": tipo_acept, "NATURALEZA_COLE": nat_cole,
            "NIVEL_PREGRADO": int(nivel_pregrado),
            "EDAD": edad,
            "ESTRATO": np.nan if sin_estrato else float(estrato),
            "ESTRATO_FALTANTE": int(sin_estrato),
            "ANTIGUEDAD_SEMESTRES": antiguedad,
            "CREDITOS_ULTIM_SEMEST_MATRIC": creditos_ult,
            "PCT_RURAL_MUNI": pct_rural,
            "PROM_SABER_MUNI": prom_saber,
            "TASA_BENEF_MUNI": tasa_benef,
            "VIVE_FUERA_ANTIOQUIA": int(vive_fuera),
        }
        print("K.1) diccionario armado:", flush=True)
        for k, v in datos.items():
            print(f"     {k} = {v!r} ({type(v).__name__})", flush=True)

        fila_sin_orden = pd.DataFrame([datos])
        print("K.2) pd.DataFrame(...) OK, shape:", fila_sin_orden.shape, flush=True)

        fila = fila_sin_orden[CAT_FEATURES + NUM_FEATURES]
        print("L) fila reordenada OK:", flush=True)
        print(fila.dtypes.to_dict(), flush=True)

        print("M) llamando pipe.predict_proba...", flush=True)
        prob = pipe.predict_proba(fila)[0, 1]
        print(f"N) predict_proba OK -> {prob:.4f}", flush=True)

        st.metric("Probabilidad de riesgo académico", f"{prob * 100:.1f} %")
        print("O) st.metric renderizado OK", flush=True)

    except Exception as e:
        print("X) EXCEPCIÓN CAPTURADA:", flush=True)
        print(traceback.format_exc(), flush=True)
        st.error(f"⚠️ Ocurrió un error al calcular el riesgo:\n\n**{type(e).__name__}**: {e}")
        with st.expander("Detalle técnico (traceback completo)"):
            st.code(traceback.format_exc())

        mensaje_js = str(e).replace("`", "'").replace("\\", "\\\\")
        components.html(f"""
            <script>
                console.error("Error en el cálculo de riesgo: {mensaje_js}");
                window.alert("Ocurrió un error al calcular el riesgo. Revisa el mensaje en la app.");
            </script>
        """, height=0)