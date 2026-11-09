FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fleetops/ fleetops/
RUN python -m fleetops.train
EXPOSE 8000
CMD ["uvicorn", "fleetops.app:from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
