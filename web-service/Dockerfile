FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY ["predict.py", "./"]

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv && pipenv install --system --deploy

CMD ["python", "predict.py", "2023", "05"]