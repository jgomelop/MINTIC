FROM python:3.12.13-slim

# libgomp1: requerido en runtime por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Evita que Streamlit quede esperando input interactivo (prompt de email/
# telemetría) en el arranque dentro del contenedor.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY outputs/ ./outputs/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]