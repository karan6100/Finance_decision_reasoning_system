FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app/src
EXPOSE 7860

CMD ["/app/start.sh"]
