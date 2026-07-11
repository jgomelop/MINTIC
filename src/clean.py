# -*- coding: utf-8 -*-
"""Fase 2a — Limpieza de los cinco insumos -> parquet en data/processed.

Problemas de calidad tratados (documentados para el concurso):
  * URABA:        FECH_NACE en formatos mixtos (datetime y texto '13-APR-1993');
                  códigos de municipio sin prefijo de departamento (854 -> 05854);
                  estratos nulos (se conservan como NA + bandera).
  * Beneficiarios: mojibake por doble codificación ('SOÃ‘ARES' -> 'SOÑARES', se
                  corrige con ftfy); categorías inconsistentes (F/FEMENINO,
                  0/NO/1/SI, URABA/URABÁ); GRADO con tipos mixtos.
  * Censo:        códigos a texto de 5 dígitos; conteos a entero.
  * Saber 11:     puntajes en texto -> numérico; códigos a 5 dígitos.
  * DIVIPOLA:     coordenadas con coma decimal -> float.

Uso:  python src/clean.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd
from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# Fecha de referencia para calcular la edad: inicio del semestre 2026-1.
REF_DATE = pd.Timestamp("2026-02-01")


# ---------------------------------------------------------------- utilidades
def fix_text(value):
    """Corrige mojibake y espacios; deja pasar valores no-texto."""
    if isinstance(value, str):
        return ftfy.fix_text(value).strip()
    return value


def norm_cat(value):
    """Categoría canónica: sin tildes, mayúsculas, un solo espacio."""
    if not isinstance(value, str):
        return value
    return " ".join(unidecode(value).upper().split())


def dane5(depto, muni) -> str | None:
    """Código DANE de 5 dígitos a partir de código de depto y de municipio."""
    try:
        d, m = int(depto), int(muni)
    except (TypeError, ValueError):
        return None
    if d <= 0 or m < 0:
        return None
    return f"{d:02d}{m:03d}"


def stringify_mixed_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a texto las columnas objeto con tipos mezclados (parquet-safe)."""
    for col in df.columns[df.dtypes == object]:
        tipos = df[col].dropna().map(type).unique()
        if len(tipos) > 1 or (len(tipos) == 1 and tipos[0] is not str):
            df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v))
    return df


def parse_fecha_mixta(series: pd.Series) -> pd.Series:
    """Parsea FECH_NACE con datetimes reales y textos tipo '13-APR-1993'."""
    ya_fecha = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    texto = series[ya_fecha.isna()].astype(str)
    extra = pd.to_datetime(texto, errors="coerce", format="%d-%b-%Y")
    return ya_fecha.fillna(extra)


# ---------------------------------------------------------------- 1. URABA
def clean_uraba() -> pd.DataFrame:
    df = pd.read_excel(RAW / "URABA 20261.xlsx", sheet_name="20261_RG_MAT")
    n0 = len(df)

    df["FECHA_NACIMIENTO"] = parse_fecha_mixta(df["FECH_NACE"])
    df["EDAD"] = ((REF_DATE - df["FECHA_NACIMIENTO"]).dt.days / 365.25).round(1)
    df.loc[(df["EDAD"] < 14) | (df["EDAD"] > 90), "EDAD"] = np.nan

    df["COD_DANE_NACE"] = [
        dane5(d, m) if p == "COLOMBIA" else None
        for d, m, p in zip(df["COD_DEPTO_NACE"], df["COD_MUNI_NACE"],
                           df["PAIS_NACE"].map(norm_cat))
    ]
    df["COD_DANE_VIVE"] = [
        dane5(d, m) if p == "COLOMBIA" else None
        for d, m, p in zip(df["COD_DEPTO_VIVE"], df["COD_MUNI_VIVE"],
                           df["PAIS_VIVE"].map(norm_cat))
    ]

    df["ESTRATO"] = pd.to_numeric(df["ESTRATO"], errors="coerce").astype("Int64")
    df["ESTRATO_FALTANTE"] = df["ESTRATO"].isna().astype(int)

    for col in ["PROMEDIO_SEMESTRE", "PROMEDIO_PROGRAMA", "PROMEDIO_UNIVERSIDAD",
                "PERIODOS_PRUEBA_PROGRAMA", "CREDGRADO", "CREDAPROBADOS",
                "NUMSEMESTRES", "SEMESTRE_INICIA_PROGRAMA",
                "CREDITOS_ULTIM_SEMEST_MATRIC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Antigüedad en semestres (1 = primer semestre en 2026-1).
    y = (df["SEMESTRE_INICIA_PROGRAMA"] // 10).astype("Int64")
    s = (df["SEMESTRE_INICIA_PROGRAMA"] % 10).astype("Int64")
    df["ANTIGUEDAD_SEMESTRES"] = ((2026 * 2 + 1) - (y * 2 + s) + 1).astype("Int64")
    df.loc[(df["ANTIGUEDAD_SEMESTRES"] < 1) |
           (df["ANTIGUEDAD_SEMESTRES"] > 40), "ANTIGUEDAD_SEMESTRES"] = pd.NA

    for col in ["SEXO", "SEDE", "FACULTAD", "PROGRAMA", "TIPO_ACEPTACION",
                "NATURALEZA_COLE", "NIVEL_PREGRADO", "NOMBRE_MUNI_NACE",
                "NOMBRE_MUNI_VIVE", "NOMBRE_COLE", "DEPARTAMENTO_VIVE"]:
        df[col] = df[col].map(fix_text).map(norm_cat)

    assert len(df) == n0, "clean_uraba alteró el número de filas"
    df = df.drop(columns=["FECH_NACE"])
    df = stringify_mixed_object_cols(df)
    df.to_parquet(PROC / "uraba_clean.parquet", index=False)
    print(f"URABA: {len(df)} filas | fechas no parseadas: "
          f"{df['FECHA_NACIMIENTO'].isna().sum()} | "
          f"sin cod DANE vive: {df['COD_DANE_VIVE'].isna().sum()} | "
          f"estrato faltante: {df['ESTRATO_FALTANTE'].sum()}")
    return df


# ---------------------------------------------------------- 2. Beneficiarios
def clean_beneficiarios() -> pd.DataFrame:
    df = pd.read_excel(RAW / "Beneficiarios.xlsx",
                       sheet_name="Beneficiarios_de_los_programas_")
    n0 = len(df)

    df.columns = [norm_cat(fix_text(c)) for c in df.columns]
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].map(fix_text)

    df["GENERO"] = df["GENERO"].map(norm_cat).map(
        {"F": "F", "FEMENINO": "F", "M": "M", "MASCULINO": "M",
         "ND": "ND", "OTRO": "OTRO"}).fillna("ND")

    df["VICTIMA_CONFLICTO"] = (
        df["VICTIMA DEL CONFLICTO ARMADO"].astype(str).str.strip().str.upper()
        .map({"0": 0, "NO": 0, "1": 1, "SI": 1, "SÍ": 1})
    )

    for col in ["SUBREGION DE RESIDENCIA", "MUNICIPIO DE RESIDENCIA",
                "MUNICIPIO DE LA INSTITUCION EDUCATIVA", "INSTITUCION EDUCATIVA",
                "PROGRAMA"]:
        df[col] = df[col].map(norm_cat)

    df["GRADO"] = (df["GRADO"].astype(str).str.strip().str.upper()
                   .replace({"NAN": "ND", "NONE": "ND"}))
    df.loc[~df["GRADO"].isin(
        ["8", "9", "10", "11", "EGRESADO", "GRADUADO", "ND"]), "GRADO"] = "ND"

    df["COD_DANE"] = (pd.to_numeric(df["CODIGO MUNICIPIO"], errors="coerce")
                      .astype("Int64").astype(str).str.zfill(5)
                      .replace("<NA>0", None))
    df.loc[~df["COD_DANE"].str.fullmatch(r"\d{5}", na=False), "COD_DANE"] = None

    df["CONVOCATORIA"] = pd.to_numeric(df["CONVOCATORIA"], errors="coerce")

    assert len(df) == n0, "clean_beneficiarios alteró el número de filas"
    df = stringify_mixed_object_cols(df)
    df.to_parquet(PROC / "beneficiarios_clean.parquet", index=False)
    print(f"Beneficiarios: {len(df)} filas | género: {dict(df['GENERO'].value_counts())} | "
          f"víctima nula: {df['VICTIMA_CONFLICTO'].isna().sum()} | "
          f"sin cod DANE: {df['COD_DANE'].isna().sum()}")
    return df


# ------------------------------------------------------------------ 3. Censo
def clean_censo() -> pd.DataFrame:
    df = pd.read_excel(RAW / "Población antioquia.xlsx",
                       sheet_name="Población_Antioquia_censada_201")
    n0 = len(df)

    df["COD_DANE"] = (pd.to_numeric(df["CodigoMunicipio"], errors="coerce")
                      .astype("Int64").astype(str).str.zfill(5))
    df["EDAD"] = pd.to_numeric(
        df["Edad"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    conteos = ["Hombres_Cabecera", "Mujeres_Cabecera", "Hombres_CentroPoblado",
               "Mujeres_CentroPoblado", "HombresRuralDisperso",
               "MujeresRuralDisperso"]
    for col in conteos:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["NOMBRE_MUNI"] = df["NombreMunicipio"].map(fix_text).map(norm_cat)
    df["POB_TOTAL_FILA"] = df[conteos].sum(axis=1)
    df["POB_HOMBRES"] = df[[c for c in conteos if "Hombres" in c or c.startswith("Hombres")]].sum(axis=1)
    df["POB_MUJERES"] = df[[c for c in conteos if "Mujeres" in c]].sum(axis=1)
    df["POB_RURAL"] = df[["Hombres_CentroPoblado", "Mujeres_CentroPoblado",
                          "HombresRuralDisperso", "MujeresRuralDisperso"]].sum(axis=1)

    assert len(df) == n0, "clean_censo alteró el número de filas"
    df.to_parquet(PROC / "censo_clean.parquet", index=False)
    print(f"Censo: {len(df)} filas | municipios: {df['COD_DANE'].nunique()} | "
          f"edades no numéricas: {df['EDAD'].isna().sum()}")
    return df


# --------------------------------------------------------------- 4. Saber 11
def clean_saber11() -> pd.DataFrame:
    df = pd.read_csv(RAW / "saber11_antioquia_2018plus.csv", dtype=str)
    n0 = len(df)

    df["COD_DANE_COLE"] = (pd.to_numeric(df["cole_cod_mcpio_ubicacion"],
                                         errors="coerce")
                           .astype("Int64").astype(str).str.zfill(5))
    for col in ["punt_global", "punt_matematicas", "punt_lectura_critica"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["periodo"] = pd.to_numeric(df["periodo"], errors="coerce").astype("Int64")
    df["ANIO"] = (df["periodo"] // 10).astype("Int64")
    df["cole_naturaleza"] = df["cole_naturaleza"].map(norm_cat)
    df["estu_genero"] = df["estu_genero"].map(norm_cat)

    assert len(df) == n0, "clean_saber11 alteró el número de filas"
    df.to_parquet(PROC / "saber11_clean.parquet", index=False)
    print(f"Saber11: {len(df)} filas | municipios: {df['COD_DANE_COLE'].nunique()} | "
          f"punt_global nulo: {df['punt_global'].isna().sum()}")
    return df


# --------------------------------------------------------------- 5. DIVIPOLA
def clean_divipola() -> pd.DataFrame:
    df = pd.read_csv(RAW / "divipola_municipios.csv", dtype=str)
    n0 = len(df)

    df["COD_DANE"] = df["cod_mpio"].str.strip().str.zfill(5)
    for col in ["longitud", "latitud"]:
        df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False),
                                errors="coerce")
    df["NOM_MPIO"] = df["nom_mpio"].map(fix_text).map(norm_cat)
    df["DPTO"] = df["dpto"].map(fix_text).map(norm_cat)

    assert len(df) == n0, "clean_divipola alteró el número de filas"
    df.to_parquet(PROC / "divipola_clean.parquet", index=False)
    print(f"DIVIPOLA: {len(df)} filas | Antioquia: {(df['cod_dpto'] == '05').sum()}")
    return df


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    clean_uraba()
    clean_beneficiarios()
    clean_censo()
    clean_saber11()
    clean_divipola()
    print("Limpieza completa -> data/processed/*.parquet")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
