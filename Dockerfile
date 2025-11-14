# Dockerfile for Power Theft Detection Streamlit app
FROM python:3.11-slim

# set workdir
WORKDIR /app

# avoid buffering
ENV PYTHONUNBUFFERED=1

# create non-root user
RUN useradd -m appuser
# copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc git && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get remove -y build-essential gcc && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# copy app
COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
