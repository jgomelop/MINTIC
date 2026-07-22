# -*- coding: utf-8 -*-
"""Punto de entrada de la app Streamlit — alterna entre módulos A y B."""
from __future__ import annotations

import streamlit as st

import modulo_a
import modulo_b

st.set_page_config(
    page_title="Alerta temprana y brechas educativas — Antioquia",
    page_icon="🎓",
    layout="wide",
)

st.title("Del colegio a la universidad")

st.sidebar.title("Navegación")
seccion = st.sidebar.radio(
    "Módulo",
    ["Módulo B · Riesgo académico", "Módulo A · Brechas municipales"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Los datos individuales de estudiantes son anónimos y agregados. "
    "Este tablero no debe usarse como único criterio para decisiones "
    "sobre una persona: es un apoyo estadístico para bienestar universitario."
)

if seccion.startswith("Módulo B"):
    modulo_b.render()
else:
    modulo_a.render()