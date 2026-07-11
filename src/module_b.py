# -*- coding: utf-8 -*-
"""Fase 4 — Módulo B: modelo predictivo de riesgo académico (alerta temprana).

Entradas:  data/processed/matriz_estudiantes.parquet
Salidas (outputs/):
  * tabla_metricas.csv           comparación de modelos (CV 5-fold y held-out)
  * fig_comparacion_modelos.png  PR-AUC de validación cruzada por modelo
  * fig_shap_summary.png         beeswarm SHAP del mejor modelo
  * importancia_shap.csv         importancia media |SHAP| por variable
  * scoring_estudiantes.csv      probabilidad de riesgo por estudiante (anónimo)
  * modelo_riesgo.pkl            pipeline completo serializado

Definiciones:
  * Población: estudiantes con historia académica (NUMSEMESTRES > 1 y promedio
    real; el valor 9.99 de PROMEDIO_PROGRAMA es un centinela de 'sin historia').
  * Target primario RIESGO = PROMEDIO_PROGRAMA < 3.0  o  PERIODOS_PRUEBA >= 1.
  * Target alternativo (sensibilidad) = rezago: créditos aprobados < 75% de lo
    esperado según semestres cursados (plan de referencia de 10 semestres).
  * Sin fuga: promedios, períodos de prueba y créditos aprobados NO son features.

Uso:  python src/module_b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
SEED = 42

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                              "#898781", "#e1e0d9", "#c3c2b7")
AZUL, AZUL_CLARO = "#2a78d6", "#9ec5f4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "Segoe UI", "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

CAT_FEATURES = ["SEXO", "SEDE", "FACULTAD", "TIPO_ACEPTACION",
                "NATURALEZA_COLE", "NIVEL_PREGRADO"]
NUM_FEATURES = ["EDAD", "ESTRATO", "ESTRATO_FALTANTE", "ANTIGUEDAD_SEMESTRES",
                "CREDITOS_ULTIM_SEMEST_MATRIC", "PCT_RURAL_MUNI",
                "PROM_SABER_MUNI", "TASA_BENEF_MUNI", "VIVE_FUERA_ANTIOQUIA"]


def cargar() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    e = pd.read_parquet(PROC / "matriz_estudiantes.parquet")
    n0 = len(e)
    e = e[(e["PROMEDIO_PROGRAMA"] <= 5) & (e["NUMSEMESTRES"] > 1)].copy()
    print(f"Población: {len(e)} de {n0} matriculados "
          f"(se excluyen {n0 - len(e)} sin historia académica)")

    y = ((e["PROMEDIO_PROGRAMA"] < 3.0)
         | (e["PERIODOS_PRUEBA_PROGRAMA"] >= 1)).astype(int)
    avance = e["CREDAPROBADOS"] / e["CREDGRADO"]
    esperado = e["NUMSEMESTRES"] / 10
    y_alt = ((avance / esperado) < 0.75).astype(int)
    print(f"Target primario (prom<3.0 o prueba): {y.mean() * 100:.1f}% "
          f"({y.sum()} casos) | alternativo (rezago): {y_alt.mean() * 100:.1f}%")

    e["ESTRATO"] = e["ESTRATO"].astype("float64")
    return e, y, y_alt


def hacer_pipeline(clf, escalar: bool) -> Pipeline:
    num_steps = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        num_steps.append(("escalar", StandardScaler()))
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
         CAT_FEATURES),
        ("num", Pipeline(num_steps), NUM_FEATURES),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def modelos() -> dict[str, Pipeline]:
    pos_weight = None  # LightGBM usa is_unbalance
    return {
        "Regresión logística": hacer_pipeline(
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED), escalar=True),
        "Random Forest": hacer_pipeline(
            RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                   class_weight="balanced_subsample",
                                   random_state=SEED, n_jobs=-1),
            escalar=False),
        "LightGBM": hacer_pipeline(
            LGBMClassifier(n_estimators=600, learning_rate=0.05,
                           num_leaves=31, min_child_samples=25,
                           is_unbalance=True, random_state=SEED,
                           verbosity=-1, n_jobs=-1), escalar=False),
    }


def comparar_cv(X_tr, y_tr) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    filas = []
    for nombre, pipe in modelos().items():
        res = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=1,
                             scoring={"roc_auc": "roc_auc",
                                      "pr_auc": "average_precision",
                                      "f1": "f1"})
        filas.append({
            "modelo": nombre,
            "cv_roc_auc": res["test_roc_auc"].mean(),
            "cv_roc_auc_std": res["test_roc_auc"].std(),
            "cv_pr_auc": res["test_pr_auc"].mean(),
            "cv_pr_auc_std": res["test_pr_auc"].std(),
            "cv_f1": res["test_f1"].mean(),
        })
        print(f"  {nombre}: ROC-AUC {filas[-1]['cv_roc_auc']:.3f} | "
              f"PR-AUC {filas[-1]['cv_pr_auc']:.3f} | F1 {filas[-1]['cv_f1']:.3f}")
    return pd.DataFrame(filas)


def fig_comparacion(tabla: pd.DataFrame, mejor: str) -> None:
    t = tabla.sort_values("cv_pr_auc")
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    colores = [AZUL if m == mejor else AZUL_CLARO for m in t["modelo"]]
    ax.barh(t["modelo"], t["cv_pr_auc"], xerr=t["cv_pr_auc_std"],
            color=colores, height=0.55, error_kw=dict(ecolor=INK_2, lw=1.2))
    for i, (v, m) in enumerate(zip(t["cv_pr_auc"], t["modelo"])):
        ax.text(v + t["cv_pr_auc_std"].max() + 0.005, i, f"{v:.3f}",
                va="center", fontsize=9,
                color=INK if m == mejor else INK_2)
    ax.set_xlabel("PR-AUC (validación cruzada 5-fold, media ± d.e.)")
    ax.set_title("Comparación de modelos — riesgo académico", loc="left",
                 color=INK)
    ax.set_xlim(0, min(1.0, t["cv_pr_auc"].max() + 0.12))
    fig.tight_layout()
    fig.savefig(OUT / "fig_comparacion_modelos.png", dpi=150)
    plt.close(fig)


def explicar_shap(pipe: Pipeline, X_bg: pd.DataFrame, X_te: pd.DataFrame) -> None:
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]

    def transformar(X):
        Xt = prep.transform(X)
        return Xt.toarray() if hasattr(Xt, "toarray") else Xt

    Xt_bg = transformar(X_bg.sample(min(500, len(X_bg)), random_state=SEED))
    Xt_te = transformar(X_te)
    nombres = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    # shap.Explainer elige el algoritmo según el modelo (Linear, Tree, ...).
    explainer = shap.Explainer(clf, Xt_bg, feature_names=nombres)
    exp = explainer(Xt_te)
    valores = exp.values
    if valores.ndim == 3:                  # (n, features, clases) -> clase 1
        exp = exp[:, :, 1]
        valores = exp.values
    plt.figure()
    shap.plots.beeswarm(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_facecolor(SURFACE)
    fig.suptitle("¿Qué empuja el riesgo académico? (SHAP, conjunto de prueba)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    imp = (pd.DataFrame({"variable": nombres,
                         "importancia_media_abs_shap": np.abs(valores).mean(0)})
           .sort_values("importancia_media_abs_shap", ascending=False))
    imp.to_csv(OUT / "importancia_shap.csv", index=False, encoding="utf-8-sig")
    print("Top 8 variables (|SHAP| medio):")
    print(imp.head(8).to_string(index=False))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    e, y, y_alt = cargar()
    X = e[CAT_FEATURES + NUM_FEATURES]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train: {len(X_tr)} | Test (held-out): {len(X_te)}")

    print("Validación cruzada (train):")
    tabla = comparar_cv(X_tr, y_tr)
    mejor = tabla.loc[tabla["cv_pr_auc"].idxmax(), "modelo"]
    print(f"Mejor modelo por PR-AUC: {mejor}")

    pipe = modelos()[mejor].fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    held = {"roc_auc": roc_auc_score(y_te, prob),
            "pr_auc": average_precision_score(y_te, prob),
            "f1": f1_score(y_te, pred)}
    print(f"Held-out — ROC-AUC {held['roc_auc']:.3f} | "
          f"PR-AUC {held['pr_auc']:.3f} | F1@0.5 {held['f1']:.3f}")
    print(classification_report(y_te, pred,
                                target_names=["sin riesgo", "riesgo"]))

    # Lift del decil superior: insight accionable para bienestar universitario.
    corte = np.quantile(prob, 0.9)
    top = prob >= corte
    captura = y_te[top].sum() / y_te.sum()
    lift = (y_te[top].mean()) / y_te.mean()
    print(f"Decil de mayor riesgo: captura {captura * 100:.0f}% de los casos "
          f"reales (lift {lift:.1f}x)")

    # Sensibilidad: mismo pipeline sobre el target alternativo de rezago.
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X, y_alt, test_size=0.2, stratify=y_alt, random_state=SEED)
    pipe_alt = modelos()[mejor].fit(Xa_tr, ya_tr)
    prob_alt = pipe_alt.predict_proba(Xa_te)[:, 1]
    print(f"Sensibilidad (target rezago): ROC-AUC "
          f"{roc_auc_score(ya_te, prob_alt):.3f} | PR-AUC "
          f"{average_precision_score(ya_te, prob_alt):.3f}")

    tabla_out = tabla.copy()
    for k, v in held.items():
        tabla_out.loc[tabla_out["modelo"] == mejor, f"heldout_{k}"] = v
    tabla_out.to_csv(OUT / "tabla_metricas.csv", index=False,
                     encoding="utf-8-sig")

    fig_comparacion(tabla, mejor)
    explicar_shap(pipe, X_tr, X_te)

    scoring = e[["SEDE", "FACULTAD", "NOMBRE_MUNI_VIVE", "PROGRAMA"]].copy()
    scoring["PROB_RIESGO"] = pipe.predict_proba(X)[:, 1].round(4)
    scoring["DECIL_RIESGO"] = pd.qcut(scoring["PROB_RIESGO"], 10,
                                      labels=False, duplicates="drop") + 1
    scoring = scoring.sort_values("PROB_RIESGO", ascending=False)
    scoring.to_csv(OUT / "scoring_estudiantes.csv", index=False,
                   encoding="utf-8-sig")

    joblib.dump(pipe, OUT / "modelo_riesgo.pkl")
    print("Módulo B completo -> outputs/")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
