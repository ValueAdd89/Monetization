from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def run_pipeline():
    print("Running monetization pipeline...")

with DAG(dag_id="monetization_dag", start_date=datetime(2023, 1, 1), schedule_interval="@daily", catchup=False) as dag:
    task = PythonOperator(task_id="run_pipeline", python_callable=run_pipeline)
