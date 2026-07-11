# -*- coding: utf-8 -*-
"""Fase 1 — Adquisición de datos abiertos desde datos.gov.co (API Socrata / SODA).

Descarga y guarda en data/raw:
  1. DIVIPOLA códigos de municipios (DANE)      -> divipola_municipios.csv
     https://www.datos.gov.co/d/gdxc-w37w
  2. Resultados únicos Saber 11 (ICFES),        -> saber11_antioquia_2018plus.csv
     microdatos filtrados a colegios de Antioquia (depto 05) y periodo >= 2018-1,
     con columnas mínimas necesarias (~298.000 filas).
     https://www.datos.gov.co/d/kgxf-xxbe

Nota de diseño: el agregado municipal se calcula localmente (src/integrate.py).
Se intentó agregar en el servidor con SoQL (avg(punt_global::number) GROUP BY
municipio), pero el query coordinator excede el tiempo de respuesta sobre las
7,1 millones de filas; descargar el microdato filtrado es robusto y auditable.

Uso:  python src/acquire.py
"""
from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.datos.gov.co/resource"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DIVIPOLA_ID = "gdxc-w37w"
SABER11_ID = "kgxf-xxbe"

SABER11_COLS = [
    "periodo",
    "cole_cod_mcpio_ubicacion",
    "cole_mcpio_ubicacion",
    "cole_naturaleza",
    "estu_cod_reside_mcpio",
    "estu_genero",
    "punt_global",
    "punt_matematicas",
    "punt_lectura_critica",
]
SABER11_WHERE = "cole_cod_depto_ubicacion = '05' AND periodo >= '20181'"
PAGE_SIZE = 100_000


def _get(url: str, params: dict, timeout: int = 300, retries: int = 3) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.RequestException,) as exc:
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"  intento {attempt} falló ({exc}); reintentando en {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_divipola() -> Path:
    out = RAW_DIR / "divipola_municipios.csv"
    print(f"[1/2] DIVIPOLA ({DIVIPOLA_ID}) -> {out.name}")
    r = _get(f"{BASE}/{DIVIPOLA_ID}.csv", {"$limit": 5000, "$order": "cod_mpio"})
    out.write_bytes(r.content)
    n = sum(1 for _ in io.StringIO(r.text)) - 1
    print(f"  {n} filas")
    return out


def download_saber11() -> Path:
    out = RAW_DIR / "saber11_antioquia_2018plus.csv"
    print(f"[2/2] Saber 11 ({SABER11_ID}) filtrado Antioquia 2018+ -> {out.name}")

    r = _get(f"{BASE}/{SABER11_ID}.json",
             {"$select": "count(*) AS n", "$where": SABER11_WHERE})
    total = int(r.json()[0]["n"])
    print(f"  total esperado: {total} filas")

    written = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = None
        offset = 0
        while True:
            r = _get(f"{BASE}/{SABER11_ID}.csv", {
                "$select": ",".join(SABER11_COLS),
                "$where": SABER11_WHERE,
                "$order": ":id",
                "$limit": PAGE_SIZE,
                "$offset": offset,
            })
            rows = list(csv.reader(io.StringIO(r.text)))
            if not rows or len(rows) <= 1:
                break
            header, data = rows[0], rows[1:]
            if writer is None:
                writer = csv.writer(fh)
                writer.writerow(header)
            writer.writerows(data)
            written += len(data)
            print(f"  página offset={offset}: +{len(data)} (acum. {written})")
            offset += PAGE_SIZE
            if len(data) < PAGE_SIZE:
                break

    if written != total:
        print(f"  ADVERTENCIA: se escribieron {written} filas, se esperaban {total}")
    else:
        print(f"  OK: {written} filas (coincide con count del servidor)")
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_divipola()
    download_saber11()
    print("Adquisición completa.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
