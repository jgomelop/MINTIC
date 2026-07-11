# -*- coding: utf-8 -*-
"""Fase 2b — Integración: homologación por código DANE y construcción de matrices.

Salidas en data/processed:
  * matriz_municipal.parquet  (125 municipios de Antioquia x ~19 variables)
  * matriz_estudiantes.parquet (8.157 matriculados x features + columnas target)
y en outputs/:
  * log_integracion.txt (conteos de control y auditoría fuzzy de nombres)

Llave de integración: código DANE de municipio (5 dígitos). La auditoría con
rapidfuzz compara nombres de municipio entre fuentes para validar el cruce.

Uso:  python src/integrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG.append(msg)


# ------------------------------------------------------- agregados por fuente
def censo_municipal(censo: pd.DataFrame) -> pd.DataFrame:
    g = censo.groupby("COD_DANE")
    base = g.agg(NOMBRE_MUNI=("NOMBRE_MUNI", "first"),
                 POB_TOTAL=("POB_TOTAL_FILA", "sum"),
                 POB_RURAL=("POB_RURAL", "sum"),
                 POB_MUJERES=("POB_MUJERES", "sum"))

    j1519 = censo[censo["EDAD"].between(15, 19)].groupby("COD_DANE")
    j1524 = censo[censo["EDAD"].between(15, 24)].groupby("COD_DANE")
    base["POB_15_19"] = j1519["POB_TOTAL_FILA"].sum()
    base["POB_15_24"] = j1524["POB_TOTAL_FILA"].sum()
    base["PCT_MUJERES_15_24"] = (j1524["POB_MUJERES"].sum()
                                 / base["POB_15_24"] * 100)
    base["PCT_RURAL"] = base["POB_RURAL"] / base["POB_TOTAL"] * 100
    return base.drop(columns=["POB_RURAL", "POB_MUJERES"]).reset_index()


def uraba_municipal(uraba: pd.DataFrame) -> pd.DataFrame:
    vive = uraba.groupby("COD_DANE_VIVE").agg(
        N_MATRIC_VIVE=("SEXO", "size"),
        PCT_MUJERES_MATRIC=("SEXO", lambda s: (s == "FEME").mean() * 100),
    ).rename_axis("COD_DANE").reset_index()
    nace = uraba.groupby("COD_DANE_NACE").agg(
        N_MATRIC_NACE=("SEXO", "size"),
    ).rename_axis("COD_DANE").reset_index()
    return vive.merge(nace, on="COD_DANE", how="outer")


def beneficiarios_municipal(benef: pd.DataFrame) -> pd.DataFrame:
    return benef.groupby("COD_DANE").agg(
        N_BENEF=("PROGRAMA", "size"),
        PCT_VICTIMAS_BENEF=("VICTIMA_CONFLICTO", lambda s: s.mean() * 100),
    ).reset_index()


def saber_municipal(saber: pd.DataFrame) -> pd.DataFrame:
    g = saber.groupby("COD_DANE_COLE")
    base = g.agg(PROM_GLOBAL_SABER=("punt_global", "mean"),
                 PROM_MATEMATICAS_SABER=("punt_matematicas", "mean"),
                 PROM_LECTURA_SABER=("punt_lectura_critica", "mean"),
                 N_EVALUADOS=("punt_global", "size"),
                 N_PERIODOS=("periodo", "nunique"))
    base["EVALUADOS_POR_PERIODO"] = base["N_EVALUADOS"] / base["N_PERIODOS"]

    nat = (saber.pivot_table(index="COD_DANE_COLE", columns="cole_naturaleza",
                             values="punt_global", aggfunc="mean"))
    if {"OFICIAL", "NO OFICIAL"} <= set(nat.columns):
        base["BRECHA_OFICIAL_SABER"] = nat["NO OFICIAL"] - nat["OFICIAL"]
    else:
        base["BRECHA_OFICIAL_SABER"] = np.nan

    gen = (saber.pivot_table(index="COD_DANE_COLE", columns="estu_genero",
                             values="punt_global", aggfunc="mean"))
    if {"F", "M"} <= set(gen.columns):
        base["BRECHA_GENERO_SABER"] = gen["F"] - gen["M"]
    else:
        base["BRECHA_GENERO_SABER"] = np.nan

    return base.drop(columns=["N_PERIODOS"]).rename_axis("COD_DANE").reset_index()


# ------------------------------------------------------------------ auditoría
def auditoria_fuzzy(matriz: pd.DataFrame, divipola: pd.DataFrame) -> None:
    """Valida que el nombre censal y el DIVIPOLA coincidan para cada código."""
    div = divipola.set_index("COD_DANE")["NOM_MPIO"]
    peor: list[tuple[str, str, str, float]] = []
    for _, row in matriz.iterrows():
        nom_div = div.get(row["COD_DANE"])
        if nom_div is None or pd.isna(row["NOMBRE_MUNI"]):
            continue
        score = fuzz.token_sort_ratio(row["NOMBRE_MUNI"], nom_div)
        if score < 85:
            peor.append((row["COD_DANE"], row["NOMBRE_MUNI"], nom_div, score))
    log(f"Auditoría fuzzy censo vs DIVIPOLA: {len(matriz) - len(peor)}/"
        f"{len(matriz)} nombres coinciden (score>=85)")
    for cod, a, b, s in peor:
        log(f"  revisar {cod}: censo='{a}' divipola='{b}' (score {s:.0f})")


# ----------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(exist_ok=True)
    uraba = pd.read_parquet(PROC / "uraba_clean.parquet")
    benef = pd.read_parquet(PROC / "beneficiarios_clean.parquet")
    censo = pd.read_parquet(PROC / "censo_clean.parquet")
    saber = pd.read_parquet(PROC / "saber11_clean.parquet")
    divipola = pd.read_parquet(PROC / "divipola_clean.parquet")

    # ------------------------------------------------ matriz municipal (A)
    m = censo_municipal(censo)                       # espina: 125 municipios
    n_censo = len(m)
    m = (m.merge(uraba_municipal(uraba), on="COD_DANE", how="left")
          .merge(beneficiarios_municipal(benef), on="COD_DANE", how="left")
          .merge(saber_municipal(saber), on="COD_DANE", how="left")
          .merge(divipola[["COD_DANE", "NOM_MPIO", "longitud", "latitud"]],
                 on="COD_DANE", how="left"))
    assert len(m) == n_censo, "el merge municipal duplicó filas"

    # Ausencia de matriculados/beneficiarios es un cero real, no un faltante.
    for col in ["N_MATRIC_VIVE", "N_MATRIC_NACE", "N_BENEF"]:
        m[col] = m[col].fillna(0).astype(int)

    m["TASA_MATRIC_VIVE"] = m["N_MATRIC_VIVE"] / m["POB_15_24"] * 1000
    m["TASA_MATRIC_NACE"] = m["N_MATRIC_NACE"] / m["POB_15_24"] * 1000
    m["TASA_BENEF"] = m["N_BENEF"] / m["POB_15_19"] * 1000
    m["BRECHA_GENERO_MATRIC"] = (m["PCT_MUJERES_MATRIC"]
                                 - m["PCT_MUJERES_15_24"])

    m.to_parquet(PROC / "matriz_municipal.parquet", index=False)
    m.to_csv(OUT / "matriz_municipal.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------ conteos de control
    log(f"Matriz municipal: {len(m)} municipios x {m.shape[1]} columnas")
    ant = uraba["COD_DANE_VIVE"].str.startswith("05", na=False)
    en_spine = uraba["COD_DANE_VIVE"].isin(set(m["COD_DANE"]))
    log(f"URABA: {ant.sum()}/{len(uraba)} residen en Antioquia; "
        f"{en_spine.sum()} ({en_spine.sum() / ant.sum() * 100:.1f}% de los de "
        f"Antioquia) cruzan con la matriz municipal")
    b_ok = benef["COD_DANE"].isin(set(m["COD_DANE"]))
    log(f"Beneficiarios: {b_ok.sum()}/{len(benef)} "
        f"({b_ok.mean() * 100:.1f}%) cruzan con la matriz municipal")
    s_ok = saber["COD_DANE_COLE"].isin(set(m["COD_DANE"]))
    log(f"Saber11: {s_ok.sum()}/{len(saber)} ({s_ok.mean() * 100:.1f}%) "
        f"cruzan con la matriz municipal")
    log(f"Municipios sin dato Saber11: "
        f"{m['PROM_GLOBAL_SABER'].isna().sum()}")
    auditoria_fuzzy(m, divipola)

    # ---------------------------------------------- matriz de estudiantes (B)
    contexto = m[["COD_DANE", "PCT_RURAL", "PROM_GLOBAL_SABER", "TASA_BENEF"]]
    e = uraba.merge(contexto, left_on="COD_DANE_VIVE", right_on="COD_DANE",
                    how="left").drop(columns=["COD_DANE"])
    e = e.rename(columns={"PCT_RURAL": "PCT_RURAL_MUNI",
                          "PROM_GLOBAL_SABER": "PROM_SABER_MUNI",
                          "TASA_BENEF": "TASA_BENEF_MUNI"})
    e["VIVE_FUERA_ANTIOQUIA"] = (~e["COD_DANE_VIVE"]
                                 .str.startswith("05", na=True)).astype(int)
    assert len(e) == len(uraba), "el merge de estudiantes alteró filas"

    cols = [
        # features candidatas (modulo B)
        "SEXO", "EDAD", "ESTRATO", "ESTRATO_FALTANTE", "SEDE", "FACULTAD",
        "TIPO_ACEPTACION", "NATURALEZA_COLE", "NIVEL_PREGRADO",
        "ANTIGUEDAD_SEMESTRES", "CREDITOS_ULTIM_SEMEST_MATRIC",
        "PCT_RURAL_MUNI", "PROM_SABER_MUNI", "TASA_BENEF_MUNI",
        "VIVE_FUERA_ANTIOQUIA",
        # identificación de contexto (no entran al modelo)
        "PROGRAMA", "COD_DANE_VIVE", "NOMBRE_MUNI_VIVE",
        # constructoras del target (excluidas de las features)
        "PROMEDIO_PROGRAMA", "PERIODOS_PRUEBA_PROGRAMA",
        "CREDAPROBADOS", "CREDGRADO", "NUMSEMESTRES",
    ]
    e[cols].to_parquet(PROC / "matriz_estudiantes.parquet", index=False)
    log(f"Matriz de estudiantes: {len(e)} filas x {len(cols)} columnas | "
        f"sin contexto municipal: {e['PCT_RURAL_MUNI'].isna().sum()}")

    (OUT / "log_integracion.txt").write_text("\n".join(LOG), encoding="utf-8")
    print("Integración completa.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
