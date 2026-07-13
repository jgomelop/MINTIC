FROM python:3.12-slim

# libgomp1: requerido en runtime por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Evita segfaults de OpenBLAS/OpenMP: dentro de contenedores, la
# autodetección de núcleos de numpy/scipy puede fallar, y esto choca
# especialmente cuando el código corre en un hilo no-principal -- como
# hace Streamlit al ejecutar el script de la app.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY outputs/ ./outputs/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]