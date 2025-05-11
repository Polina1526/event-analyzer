FROM python:3.11-slim

RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

EXPOSE 8505

CMD ["streamlit", "run", "app/home.py"]
