# Del colegio a la universidad: inteligencia territorial para cerrar brechas de acceso y permanencia en la educación superior de Antioquia

Proyecto para el concurso **[Datos al Ecosistema 2026 — IA para Colombia](https://www.datos.gov.co/stories/s/ddau-8cy9)** (nivel intermedio).

Dos módulos de analítica sobre cinco conjuntos de datos integrados:

* **Módulo A — Tipologías de municipios e índice de brecha (clustering).** ¿En qué
  municipios de Antioquia es mayor la brecha de acceso a la educación superior?
  KMeans sobre la matriz municipal + índice compuesto 0-100 + mapa interactivo.
* **Módulo B — Riesgo académico (clasificación supervisada).** ¿Qué estudiantes
  matriculados tienen mayor probabilidad de bajo rendimiento? Comparación de
  regresión logística, Random Forest y LightGBM con validación cruzada,
  interpretabilidad SHAP y scoring de alerta temprana. Incluye una **calculadora
  interactiva** en la app que estima el riesgo de un estudiante a partir de sus
  datos de entrada (ver [La aplicación interactiva](#la-aplicación-interactiva)).

## Datos integrados (5 conjuntos, 4 de datos.gov.co)

| # | Conjunto | Fuente | Obtención |
|---|---|---|---|
| 1 | Matriculados UdeA sedes regionales 2026-1 | Universidad de Antioquia | local (`data/raw/URABA 20261.xlsx`) |
| 2 | [Beneficiarios de programas de acompañamiento](https://www.datos.gov.co/d/xk8x-i6kn) | datos.gov.co — Gobernación de Antioquia | local (`data/raw/Beneficiarios.xlsx`) |
| 3 | [Población Antioquia censada 2018](https://www.datos.gov.co/d/evm3-92yw) | datos.gov.co — Gobernación de Antioquia / DANE | local (`data/raw/Población antioquia.xlsx`) |
| 4 | [Resultados únicos Saber 11](https://www.datos.gov.co/d/kgxf-xxbe) | datos.gov.co — ICFES | API Socrata (`src/acquire.py`), filtro Antioquia 2018+ |
| 5 | [DIVIPOLA códigos de municipios](https://www.datos.gov.co/d/gdxc-w37w) | datos.gov.co — DANE | API Socrata (`src/acquire.py`) |

Llave de integración: **código DANE de municipio (5 dígitos)**, homologado contra
DIVIPOLA con auditoría difusa de nombres (rapidfuzz). Cruce logrado: **100%** en
todas las fuentes.


## Resultados principales

* **3 tipologías de municipios** (silhouette 0,32; estabilidad ARI 0,94):
  cabeceras urbanas con Saber 11 alto · rurales con Saber 11 bajo y poca matrícula ·
  municipios con sede regional UdeA y matrícula per cápita alta.
* **Ranking de brecha**: Betulia, Murindó, Ituango, Angostura y Caicedo encabezan
  la lista — alta ruralidad + bajo Saber 11 + baja matrícula + baja cobertura de
  acompañamiento: candidatos a expandir Semestre Cero / Soñares / PIES.
* **Modelo de riesgo académico** (regresión logística balanceada, mejor PR-AUC):
  ROC-AUC 0,87 en held-out; el **decil superior captura 63% de los casos reales
  (lift 6,2×)** — con capacidad para acompañar al 10% de la matrícula se concentra
  dos tercios del riesgo.
* Factores de riesgo (SHAP): antigüedad sin avance de nivel, facultad (Ingeniería,
  Educación); ser mujer aparece como factor protector; el contexto municipal
  (ruralidad, Saber 11 local) aporta señal moderada.

## Requisitos
 
- Python **3.12.13** (recomendado vía [pyenv](https://github.com/pyenv/pyenv))
- Podman o Docker, si vas a correr la app en contenedor
- Los archivos ya generados en `outputs/` (`modelo_riesgo.pkl`,
  `tabla_metricas.csv`, etc.) — corre `python src/module_b.py` y
  `python src/module_a.py` si aún no existen

## Crear el entorno virtual

### Linux / Mac

```bash
# Fijar la versión de Python del proyecto (una sola vez, con pyenv instalado)
pyenv install 3.12.13
pyenv local 3.12.13

# Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Windows (PowerShell)

```powershell
# Fijar la versión de Python del proyecto (una sola vez, con pyenv-win instalado)
pyenv install 3.12.13
pyenv local 3.12.13

# Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements-dev.txt
```
Si PowerShell bloquea la activación por política de ejecución de scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Sin pyenv (cualquier SO)

Si ya tienes Python 3.12 instalado en el sistema, puedes saltarte pyenv:

```bash
python3.12 -m venv .venv        # Linux/Mac
python3.12 -m venv .venv        # Windows: usa "py -3.12 -m venv .venv"
```

y luego activa el entorno según tu sistema operativo (comandos de arriba).

### Verificar que quedó bien

```bash
python --version    # debe mostrar Python 3.12.13
```
## Reproducción de Pipeline de Entrenamiento
Los notebooks `notebooks/01…03` cuentan la historia completa (limpieza, módulo A,
módulo B) ejecutando estas mismas funciones.

```bash
.venv\Scripts\python src/acquire.py     # descarga datasets 4 y 5 (API datos.gov.co)
.venv\Scripts\python src/clean.py       # limpieza -> data/processed/*.parquet
.venv\Scripts\python src/integrate.py   # matrices municipal y de estudiantes
.venv\Scripts\python src/module_a.py    # clustering + índice de brecha + mapa
.venv\Scripts\python src/module_b.py    # modelo de riesgo + SHAP + scoring
```


## La aplicación interactiva

La app de Streamlit (`app/main.py`) tiene dos módulos en la barra lateral. El
**Módulo B** incluye la pestaña **«Predicción individual»**: una *calculadora de
riesgo académico* que estima la probabilidad de bajo rendimiento de un estudiante
(hipotético) a partir de los valores que se ingresan en un formulario.

- **Entradas — perfil académico.** Sexo, sede, facultad, tipo de aceptación,
  naturaleza del colegio, nivel de pregrado, edad, estrato (con casilla «no
  conoce / no aplica»), antigüedad en semestres, créditos del último semestre y
  si vive fuera de Antioquia; más el **municipio de residencia**.
- **Autocompletado del contexto territorial.** Al elegir el municipio, el % rural,
  el promedio Saber 11 municipal y la tasa de acompañamiento se rellenan solos
  desde `outputs/ranking_brechas.csv` (salida del Módulo A), para no pedir datos
  que el usuario no conoce.
- **Opciones seguras.** Los campos categóricos solo ofrecen las categorías que el
  `OneHotEncoder` vio durante el entrenamiento, así el formulario nunca propone un
  valor que el modelo no conoce.
- **Salida.** Al pulsar **«Calcular riesgo»**, el pipeline serializado
  (`outputs/modelo_riesgo.pkl`) devuelve la probabilidad de riesgo, mostrada como
  porcentaje. Cuando es ≥ 50 % se despliega una **alerta**: es un apoyo
  estadístico, **no un diagnóstico individual**, y debe leerse junto con el
  acompañamiento humano de bienestar (ver la [Nota ética](#nota-ética)).

La pestaña **«Resumen del modelo»** complementa con la tabla de métricas, la
comparación de modelos y la importancia SHAP de cada variable.

Para verla, levanta la app con cualquiera de las dos opciones de abajo y abre el
**Módulo B** en la barra lateral.

## Opción A — Correr con contenedor (Podman/Docker)
 
```bash
# Construir la imagen
podman build -t riesgo-academico:dev .
 
# Correr el contenedor
podman run -p 127.0.0.1:8501:8501 --name riesgo-academico riesgo-academico:dev
```
 
Abre **http://127.0.0.1:8501** en el navegador.
 
> **Nota (Fedora/Linux):** usa `127.0.0.1` en vez de `localhost` en la URL —
> en algunos sistemas `localhost` resuelve primero a IPv6 y el mapeo de
> puertos de Podman solo aplica sobre IPv4, lo que da un error de conexión
> aunque el contenedor esté corriendo bien.
 
Detener y limpiar:
```bash
podman stop riesgo-academico
podman rm riesgo-academico
```
 
Con Docker, los mismos comandos funcionan reemplazando `podman` por `docker`.
 
## Opción B — Correr localmente con Streamlit (sin contenedor)
 
```bash
# Fijar la versión de Python del proyecto (una sola vez)
pyenv install 3.12.13   # si no la tienes instalada
pyenv local 3.12.13
 
# Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate
 
# Instalar dependencias (producción + entrenamiento + notebooks)
pip install --upgrade pip
pip install -r requirements-dev.txt
 
# Correr la app
streamlit run app/main.py
```
 
Abre **http://localhost:8501**.
 
## Reentrenar el modelo
 
```bash
python src/module_b.py   # regenera outputs/modelo_riesgo.pkl y métricas
python src/module_a.py   # regenera ranking, mapa y figuras del módulo A
```
 
Si reentrenas con una versión de `scikit-learn`/`numpy`/`scipy` distinta a
la actual, actualiza esas versiones exactas en `requirements.txt` antes de
reconstruir la imagen — un `.pkl` deserializado con una versión distinta a
la de entrenamiento puede fallar o comportarse de forma inesperada.

## Estructura

```
├─ data/raw/         insumos originales + descargas de la API
├─ data/processed/   parquet limpios y matrices analíticas
├─ src/              pipeline reproducible (acquire, clean, integrate, module_a, module_b)
├─ notebooks/        01 limpieza · 02 módulo A · 03 módulo B
├─ outputs/          mapa_brechas.html, ranking_brechas.csv, scoring_estudiantes.csv,
│                    modelo_riesgo.pkl, figuras y métricas
├─ app/              main.py, modulo_a.py, modulo_b.py (app Streamlit)
├─ Dockerfile
├─ requirements.txt       dependencias de producción (app)
└─ requirements-dev.txt   producción + entrenamiento + notebooks
```

## Nota ética

El scoring de riesgo se diseñó para **priorizar acompañamiento**, nunca para
restringir el acceso a servicios. Los datos de estudiantes están anonimizados y
el modelo hereda desigualdades históricas del territorio que el informe hace
explícitas en lugar de ocultar.
