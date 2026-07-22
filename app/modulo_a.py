# -*- coding: utf-8 -*-
"""Módulo A — Tipologías municipales y brechas de acceso (Streamlit)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

OUT = Path(__file__).resolve().parents[1] / "outputs"


@st.cache_data
def cargar_csv(nombre: str) -> pd.DataFrame | None:
    ruta = OUT / nombre
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


def render() -> None:
    st.title("Módulo A · Tipologías municipales y brechas de acceso")

    ranking = cargar_csv("ranking_brechas.csv")
    mapa = OUT / "mapa_brechas.html"

    if ranking is None and not mapa.exists():
        st.info(
            "Este módulo todavía no tiene outputs generados. Corre "
            "`python src/module_a.py` y los resultados aparecerán aquí "
            "automáticamente — no requiere cambios de código."
        )
        return

    if mapa.exists():
        st.subheader("Mapa de brechas de acceso a educación superior")
        components.html(mapa.read_text(encoding="utf-8"), height=580,
                        scrolling=False)

    col1, col2 = st.columns(2)
    fig_pca = OUT / "fig_pca_clusters.png"
    fig_perfil = OUT / "fig_perfil_clusters.png"
    if fig_pca.exists():
        col1.image(str(fig_pca), caption="Tipologías (PCA + KMeans)")
    if fig_perfil.exists():
        col2.image(str(fig_perfil), caption="Perfil z-score por tipología")

    if ranking is not None:
        st.subheader("Ranking de municipios por índice de brecha")
        muni_filtro = st.text_input("Buscar municipio")
        vista = ranking.copy()
        if muni_filtro:
            vista = vista[vista["NOMBRE_MUNI"].str.contains(
                muni_filtro.upper(), na=False)]
        st.dataframe(vista, width="stretch", height=420)
        st.download_button(
            "Descargar ranking completo", ranking.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_brechas.csv", mime="text/csv",
        )