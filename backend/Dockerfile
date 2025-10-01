FROM python:3.12

WORKDIR /app


COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt


COPY ./app ./app
COPY ./logs ./logs
COPY ./models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]