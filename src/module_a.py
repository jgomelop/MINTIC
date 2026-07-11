# -*- coding: utf-8 -*-
"""Fase 3 — Módulo A: tipologías de municipios (KMeans) e índice de brecha.

Entradas:  data/processed/matriz_municipal.parquet
Salidas (outputs/):
  * ranking_brechas.csv        ranking de municipios por índice de brecha (0-100)
  * perfil_clusters.csv        perfil z-score de cada tipología
  * mapa_brechas.html          mapa interactivo (plotly, autocontenido)
  * fig_k_seleccion.png        silhouette + codo para elegir k
  * fig_pca_clusters.png       municipios en el plano PCA coloreados por clúster
  * fig_perfil_clusters.png    heatmap z-score de perfiles

Decisiones metodológicas:
  * Variables estandarizadas (z-score); población en log10 para atenuar a Medellín.
  * Faltantes imputados con la mediana (se reporta cuántos por variable).
  * k elegido por silhouette máximo en k=2..8 (se reporta también el codo).
  * Estabilidad: ARI promedio contra 5 semillas alternativas.

Uso:  python src/module_a.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
SEED = 42

# ------------------------------------------------ paleta (dataviz validada)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
       "#e34948", "#e87ba4", "#eb6834"]                     # orden fijo
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]                # claro -> oscuro
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"  # azul-gris-rojo

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "Segoe UI", "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Variables de clustering (núcleo con baja proporción de faltantes).
# Se excluyen de la métrica de distancia — pero se conservan como contexto en
# el ranking — las variables con imputación masiva: BRECHA_OFICIAL_SABER (85/125
# municipios sin colegios no oficiales), BRECHA_GENERO_SABER y
# PCT_VICTIMAS_BENEF; imputar >50% con la mediana distorsionaría los clústeres.
FEATURES = {
    "LOG_POB_15_19":        ("población joven (log)", None),
    "PCT_RURAL":            ("% rural", True),
    "TASA_MATRIC_VIVE":     ("matrícula UdeA /1.000 jóv.", False),
    "TASA_MATRIC_NACE":     ("matrícula por origen /1.000", False),
    "TASA_BENEF":           ("acompañamiento /1.000 jóv.", False),
    "PROM_GLOBAL_SABER":    ("Saber 11 global", False),
    "BRECHA_GENERO_MATRIC": ("brecha género matrícula", None),
}
# Componentes del índice de brecha y su dirección (True = más valor, más brecha)
INDEX_PARTS = {"TASA_MATRIC_VIVE": False, "PROM_GLOBAL_SABER": False,
               "PCT_RURAL": True, "TASA_BENEF": False}


def preparar(m: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    m = m.copy()
    m["LOG_POB_15_19"] = np.log10(m["POB_15_19"].clip(lower=1))
    X = m[list(FEATURES)].copy()
    faltantes = X.isna().sum()
    for col, n in faltantes[faltantes > 0].items():
        print(f"  imputando mediana en {col}: {n} municipios")
        X[col] = X[col].fillna(X[col].median())
    Xz = StandardScaler().fit_transform(X)
    return m, Xz


def elegir_k(Xz: np.ndarray) -> tuple[int, pd.DataFrame]:
    """k por silhouette máximo con k>=3.

    k=2 maximiza el silhouette global pero reproduce únicamente la dicotomía
    urbano/rural, sin valor accionable como tipología; se reporta la tabla
    completa para transparencia.
    """
    filas = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xz)
        filas.append({"k": k, "silhouette": silhouette_score(Xz, km.labels_),
                      "inercia": km.inertia_})
    tabla = pd.DataFrame(filas)
    validos = tabla[tabla["k"] >= 3]
    mejor = int(validos.loc[validos["silhouette"].idxmax(), "k"])
    return mejor, tabla


def fig_seleccion_k(tabla: pd.DataFrame, k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.plot(tabla["k"], tabla["silhouette"], color=CAT[0], lw=2,
            marker="o", ms=6)
    fila_k = tabla.loc[tabla["k"] == k].iloc[0]
    ax.scatter([k], [fila_k["silhouette"]], s=90, color=CAT[0], zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"k = {k}", (k, fila_k["silhouette"]),
                textcoords="offset points", xytext=(8, 6), color=INK)
    ax.set_xlabel("número de clústeres (k)")
    ax.set_ylabel("silhouette")
    ax.set_title("Selección de k — silhouette", loc="left", color=INK)

    ax = axes[1]
    ax.plot(tabla["k"], tabla["inercia"], color=CAT[0], lw=2, marker="o", ms=6)
    ax.set_xlabel("número de clústeres (k)")
    ax.set_ylabel("inercia (codo)")
    ax.set_title("Método del codo", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_k_seleccion.png", dpi=150)
    plt.close(fig)


def etiquetas_clusters(perfil_z: pd.DataFrame) -> dict[int, str]:
    """Nombra cada clúster por sus dos rasgos más desviados del promedio."""
    nombres = {}
    for c in perfil_z.index:
        z = perfil_z.loc[c].drop("LOG_POB_15_19", errors="ignore")
        top = z.abs().sort_values(ascending=False).head(2).index
        partes = []
        for f in top:
            alta = z[f] > 0
            partes.append(("alta " if alta else "baja ") + FEATURES[f][0])
        nombres[c] = " y ".join(partes)
    return nombres


def fig_pca(m: pd.DataFrame, Xz: np.ndarray, nombres: dict[int, str]) -> None:
    pca = PCA(n_components=2, random_state=SEED)
    P = pca.fit_transform(Xz)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for c in sorted(m["CLUSTER"].unique()):
        sel = m["CLUSTER"] == c
        ax.scatter(P[sel, 0], P[sel, 1], s=55, color=CAT[c],
                   edgecolor=SURFACE, linewidth=1.5,
                   label=f"C{c + 1} · {nombres[c]}")
    # etiqueta directa solo para municipios notables (selectivo, no todos)
    notables = (m.assign(x=P[:, 0], y=P[:, 1])
                  .sort_values("POB_15_19", ascending=False).head(6))
    extremos = m.assign(x=P[:, 0], y=P[:, 1]).nlargest(4, "INDICE_BRECHA")
    for _, r in pd.concat([notables, extremos]).drop_duplicates("COD_DANE").iterrows():
        ax.annotate(r["NOMBRE_MUNI"].title(), (r["x"], r["y"]), fontsize=8,
                    color=INK_2, textcoords="offset points", xytext=(6, 4))
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"componente 1 ({var[0]:.0f}% de la varianza)")
    ax.set_ylabel(f"componente 2 ({var[1]:.0f}%)")
    ax.set_title("Tipologías de municipios de Antioquia (PCA + KMeans)",
                 loc="left", color=INK)
    ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pca_clusters.png", dpi=150)
    plt.close(fig)


def fig_perfiles(perfil_z: pd.DataFrame, nombres: dict[int, str]) -> None:
    cols = [c for c in perfil_z.columns if c != "LOG_POB_15_19"]
    Z = perfil_z[cols]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(Z) + 2.6))
    lim = np.nanmax(np.abs(Z.values))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", [DIV_NEG, DIV_MID, DIV_POS])
    im = ax.imshow(Z.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)),
                  [FEATURES[c][0] for c in cols], rotation=35, ha="right",
                  fontsize=8)
    ax.set_yticks(range(len(Z)),
                  [f"C{c + 1} · {nombres[c]}" for c in Z.index], fontsize=8)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if abs(v) > lim * 0.55 else INK)
    ax.grid(False)
    ax.set_title("Perfil de cada tipología (z-score vs promedio departamental)",
                 loc="left", color=INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("desviaciones estándar", color=INK_2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_perfil_clusters.png", dpi=150)
    plt.close(fig)


def indice_brecha(m: pd.DataFrame) -> pd.Series:
    partes = []
    for col, mayor_es_brecha in INDEX_PARTS.items():
        v = m[col].fillna(m[col].median())
        r = (v - v.min()) / (v.max() - v.min())
        partes.append(r if mayor_es_brecha else 1 - r)
    return (sum(partes) / len(partes) * 100).round(1)


def mapa(m: pd.DataFrame, nombres: dict[int, str]) -> None:
    d = m.dropna(subset=["latitud", "longitud"]).copy()
    d["Municipio"] = d["NOMBRE_MUNI"].str.title()
    d["Tipología"] = d["CLUSTER"].map(lambda c: f"C{c + 1} · {nombres[c]}")
    fig = px.scatter_map(
        d, lat="latitud", lon="longitud",
        color="INDICE_BRECHA", size="POB_15_19",
        color_continuous_scale=SEQ_BLUE, size_max=34, zoom=6.7,
        map_style="carto-positron",
        hover_name="Municipio",
        hover_data={"latitud": False, "longitud": False, "POB_15_19": ":,",
                    "INDICE_BRECHA": ":.1f", "TASA_MATRIC_VIVE": ":.1f",
                    "TASA_BENEF": ":.1f", "PROM_GLOBAL_SABER": ":.0f",
                    "Tipología": True},
        labels={"INDICE_BRECHA": "Índice de brecha",
                "POB_15_19": "Población 15-19",
                "TASA_MATRIC_VIVE": "Matrícula UdeA /1.000",
                "TASA_BENEF": "Acompañamiento /1.000",
                "PROM_GLOBAL_SABER": "Saber 11 global"},
        title=("Brechas de acceso a educación superior en Antioquia — "
               "tamaño: población joven; color: índice de brecha (0-100)"),
    )
    fig.update_layout(font=dict(family="Segoe UI", color=INK),
                      paper_bgcolor=SURFACE, margin=dict(l=10, r=10, t=60, b=10))
    fig.write_html(OUT / "mapa_brechas.html", include_plotlyjs=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m = pd.read_parquet(PROC / "matriz_municipal.parquet")
    print(f"Módulo A: {len(m)} municipios")

    m, Xz = preparar(m)
    k, tabla_k = elegir_k(Xz)
    print("Selección de k:\n", tabla_k.round(3).to_string(index=False))
    print(f"k elegido por silhouette: {k}")

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xz)
    m["CLUSTER"] = km.labels_

    aris = [adjusted_rand_score(
        km.labels_,
        KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Xz))
        for s in (1, 7, 13, 99, 2024)]
    print(f"Estabilidad (ARI promedio vs 5 semillas): {np.mean(aris):.3f}")

    m["INDICE_BRECHA"] = indice_brecha(m)

    perfil_z = (pd.DataFrame(Xz, columns=list(FEATURES), index=m.index)
                .groupby(m["CLUSTER"]).mean())
    nombres = etiquetas_clusters(perfil_z)
    for c, nom in nombres.items():
        sub = m[m["CLUSTER"] == c]
        print(f"  C{c + 1} ({len(sub)} municipios) {nom} | "
              f"brecha media {sub['INDICE_BRECHA'].mean():.1f}")

    fig_seleccion_k(tabla_k, k)
    fig_pca(m, Xz, nombres)
    fig_perfiles(perfil_z, nombres)
    mapa(m, nombres)

    ranking = (m[["COD_DANE", "NOMBRE_MUNI", "CLUSTER", "INDICE_BRECHA",
                  "POB_15_19", "PCT_RURAL", "TASA_MATRIC_VIVE", "TASA_BENEF",
                  "PROM_GLOBAL_SABER", "BRECHA_GENERO_MATRIC"]]
               .sort_values("INDICE_BRECHA", ascending=False))
    ranking.insert(2, "TIPOLOGIA",
                   ranking["CLUSTER"].map(lambda c: f"C{c + 1} · {nombres[c]}"))
    ranking.to_csv(OUT / "ranking_brechas.csv", index=False,
                   encoding="utf-8-sig")
    perfil_z.rename(index=lambda c: f"C{c + 1}").to_csv(
        OUT / "perfil_clusters.csv", encoding="utf-8-sig")

    m.to_parquet(PROC / "matriz_municipal_clusters.parquet", index=False)
    print("Top 10 municipios con mayor brecha:")
    print(ranking.head(10)[["NOMBRE_MUNI", "TIPOLOGIA", "INDICE_BRECHA"]]
          .to_string(index=False))
    print("Módulo A completo -> outputs/")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
