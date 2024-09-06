FROM python:3.11.7-slim

WORKDIR /app

COPY [ "models", "models" ]

RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "app.py", "app.py" ]

ENTRYPOINT [ "python", "app.py" ]
