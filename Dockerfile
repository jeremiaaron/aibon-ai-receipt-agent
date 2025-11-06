# Base image
FROM python:3.11.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 -r requirements.txt --verbose

COPY . .

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "aibon.py", "--server.port=8501", "--server.address=0.0.0.0"]
