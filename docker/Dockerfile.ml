FROM python:3.11
WORKDIR /ml
COPY . .
RUN pip install -r requirements/ml.txt
CMD ["python", "ml_models/training/churn_model.py"]
