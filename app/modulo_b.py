# -*- coding: utf-8 -*-
"""Módulo B — Alerta temprana de riesgo académico (Streamlit)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"

CAT_FEATURES = ["SEXO", "SEDE", "FACULTAD", "TIPO_ACEPTACION",
                "NATURALEZA_COLE", "NIVEL_PREGRADO"]
NUM_FEATURES = ["EDAD", "ESTRATO", "ESTRATO_FALTANTE", "ANTIGUEDAD_SEMESTRES",
                "CREDITOS_ULTIM_SEMEST_MATRIC", "PCT_RURAL_MUNI",
                "PROM_SABER_MUNI", "TASA_BENEF_MUNI", "VIVE_FUERA_ANTIOQUIA"]


@st.cache_resource
def cargar_modelo():
    ruta = OUT / "modelo_riesgo.pkl"
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_data
def cargar_csv(nombre: str) -> pd.DataFrame | None:
    ruta = OUT / nombre
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


@st.cache_data
def cargar_lookup_municipal() -> pd.DataFrame | None:
    """Reutiliza el ranking del módulo A (si existe) para autocompletar
    contexto municipal en el formulario, evitando pedirle al usuario datos
    que no conoce (tasa de beneficiarios, Saber 11 del municipio, etc.)."""
    ruta = OUT / "ranking_brechas.csv"
    if not ruta.exists():
        return None
    df = pd.read_csv(ruta)
    cols = {"NOMBRE_MUNI": "NOMBRE_MUNI", "PCT_RURAL": "PCT_RURAL_MUNI",
            "PROM_GLOBAL_SABER": "PROM_SABER_MUNI", "TASA_BENEF": "TASA_BENEF_MUNI"}
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        return None
    return df[list(cols)].rename(columns=cols)


def categorias_desde_pipeline(pipe) -> dict[str, list]:
    """Extrae las categorías vistas en entrenamiento directamente del
    OneHotEncoder ya ajustado, para que el formulario nunca ofrezca una
    opción que el modelo no conoce.

    IMPORTANTE: `ohe.categories_` son arrays de NumPy, no listas -- el
    `list(cats)` es obligatorio. Un array de NumPy como `options` de un
    `st.selectbox` causa un segfault silencioso en el segundo rerun del
    script (cuando Streamlit necesita ubicar el índice del valor ya
    seleccionado dentro de las opciones)."""
    prep = pipe.named_steps["prep"]
    ohe = prep.named_transformers_["cat"]
    return {col: list(cats) for col, cats in zip(CAT_FEATURES, ohe.categories_)}


def render() -> None:
    st.title("Módulo B · Alerta temprana de riesgo académico")

    pipe = cargar_modelo()
    if pipe is None:
        st.error(
            "No se encontró `outputs/modelo_riesgo.pkl`. Corre "
            "`python src/module_b.py` y vuelve a desplegar."
        )
        return

    # tab_resumen, tab_prediccion, tab_listado = st.tabs(
    #     ["Resumen del modelo", "Predicción individual", "Estudiantes"]
    # )
    tab_prediccion, tab_resumen,  = st.tabs(
        ["Predicción individual", "Resumen del modelo"]
    )


    with tab_prediccion:
        st.caption(
            "Las opciones de los campos categóricos se generan a partir de "
            "las categorías que el modelo vio durante el entrenamiento."
        )
        categorias = categorias_desde_pipeline(pipe)
        lookup_muni = cargar_lookup_municipal()

        with st.form("form_prediccion"):
            c1, c2 = st.columns(2)
            with c1:
                sexo = st.selectbox("Sexo", categorias["SEXO"])
                sede = st.selectbox("Sede", categorias["SEDE"])
                facultad = st.selectbox("Facultad", categorias["FACULTAD"])
                tipo_acept = st.selectbox("Tipo de aceptación",
                                          categorias["TIPO_ACEPTACION"])
                nat_cole = st.selectbox("Naturaleza del colegio",
                                        categorias["NATURALEZA_COLE"])
                nivel_pregrado = st.selectbox("Nivel de pregrado",
                                              categorias["NIVEL_PREGRADO"])
            with c2:
                edad = st.number_input("Edad", min_value=15, max_value=70, value=20)
                sin_estrato = st.checkbox("No conoce / no aplica estrato")
                estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6],
                                       disabled=sin_estrato)
                antiguedad = st.number_input("Antigüedad (semestres)",
                                             min_value=1, max_value=40, value=1)
                creditos_ult = st.number_input(
                    "Créditos matriculados último semestre",
                    min_value=0, max_value=30, value=16)
                vive_fuera = st.checkbox("Vive fuera de Antioquia")

            st.markdown("**Contexto del municipio de residencia**")
            if lookup_muni is not None:
                municipio = st.selectbox(
                    "Municipio", sorted(lookup_muni["NOMBRE_MUNI"].unique()))
                fila_muni = lookup_muni[
                    lookup_muni["NOMBRE_MUNI"] == municipio].iloc[0]
                pct_rural = float(fila_muni["PCT_RURAL_MUNI"])
                prom_saber = float(fila_muni["PROM_SABER_MUNI"])
                tasa_benef = float(fila_muni["TASA_BENEF_MUNI"])
                st.caption(
                    f"% rural: {pct_rural:.1f} | Saber 11 municipal: "
                    f"{prom_saber:.0f} | Acompañamiento: {tasa_benef:.1f}/1.000"
                )
            else:
                st.info(
                    "No hay `ranking_brechas.csv` del módulo A todavía: "
                    "ingresa el contexto municipal manualmente."
                )
                pct_rural = st.number_input("% población rural del municipio",
                                            0.0, 100.0, 30.0)
                prom_saber = st.number_input("Promedio Saber 11 del municipio",
                                             0.0, 500.0, 250.0)
                tasa_benef = st.number_input(
                    "Tasa de acompañamiento del municipio (/1.000 jóv.)",
                    0.0, 500.0, 10.0)

            enviado = st.form_submit_button("Calcular riesgo")

        if enviado:
            # IMPORTANTE: nivel_pregrado viene de categorias["NIVEL_PREGRADO"],
            # cuyos elementos son np.int64 (columna numérica original). Mezclar
            # un solo escalar de NumPy con puros tipos nativos de Python en el
            # diccionario, al construir un DataFrame de una fila con pandas 3.0
            # (backend Arrow) dentro del hilo de Streamlit, causa un segfault
            # silencioso -- por eso se castea explícitamente a int nativo.
            nivel_pregrado_int = (
                int(nivel_pregrado) if nivel_pregrado is not None else 0
            )
            try:
                fila = pd.DataFrame([{
                    "SEXO": sexo, "SEDE": sede, "FACULTAD": facultad,
                    "TIPO_ACEPTACION": tipo_acept, "NATURALEZA_COLE": nat_cole,
                    "NIVEL_PREGRADO": nivel_pregrado_int,
                    "EDAD": edad,
                    "ESTRATO": np.nan if sin_estrato else float(estrato),
                    "ESTRATO_FALTANTE": int(sin_estrato),
                    "ANTIGUEDAD_SEMESTRES": antiguedad,
                    "CREDITOS_ULTIM_SEMEST_MATRIC": creditos_ult,
                    "PCT_RURAL_MUNI": pct_rural,
                    "PROM_SABER_MUNI": prom_saber,
                    "TASA_BENEF_MUNI": tasa_benef,
                    "VIVE_FUERA_ANTIOQUIA": int(vive_fuera),
                }])[CAT_FEATURES + NUM_FEATURES]

                prob = pipe.predict_proba(fila)[0, 1]
                st.metric("Probabilidad de riesgo académico", f"{prob * 100:.1f} %")
                if prob >= 0.5:
                    st.warning(
                        "El modelo sugiere riesgo (promedio < 3.0 o período de "
                        "prueba). Esto es una alerta estadística, no un "
                        "diagnóstico individual: debe leerse junto con "
                        "acompañamiento humano del programa de bienestar."
                    )
                else:
                    st.success("El modelo no marca riesgo elevado para este perfil.")
            except Exception as e:
                st.error(f"⚠️ Ocurrió un error al calcular el riesgo: {type(e).__name__}: {e}")
                with st.expander("Detalle técnico"):
                    import traceback
                    st.code(traceback.format_exc())

    with tab_resumen:
        tabla = cargar_csv("tabla_metricas.csv")
        if tabla is not None:
            st.subheader("Comparación de modelos (validación cruzada)")
            st.dataframe(tabla, width="stretch")

        col1, col2 = st.columns(2)
        fig_comp = OUT / "fig_comparacion_modelos.png"
        fig_shap = OUT / "fig_shap_summary.png"
        if fig_comp.exists():
            col1.image(str(fig_comp), caption="PR-AUC por modelo (CV 5-fold)")
        if fig_shap.exists():
            col2.image(str(fig_shap), caption="Impacto de cada variable (SHAP)")

        imp = cargar_csv("importancia_shap.csv")
        if imp is not None:
            st.subheader("Importancia media |SHAP| por variable")
            st.bar_chart(imp.set_index("variable").head(12))