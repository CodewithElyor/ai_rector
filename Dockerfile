# Streamlit + ETL uchun umumiy image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python kutubxonalarini o‘rnatamiz
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Loyiha fayllarini konteynerga nusxalaymiz
COPY . /app

# Streamlit port
EXPOSE 8501

# Default: dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
