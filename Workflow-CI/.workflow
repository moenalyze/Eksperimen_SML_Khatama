name: CI Pipeline & Docker Push

on:
  push:
    paths:
      - 'Workflow-CI/**'
  workflow_dispatch:

env:
  DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
  DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
  DOCKER_USERNAME: ${{ secrets.DOCKER_HUB_USERNAME }}
  DOCKER_PASSWORD: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
  MLFLOW_TRACKING_URI: https://dagshub.com/${{ secrets.DAGSHUB_USERNAME }}/Eksperimen_SML_Khatama.mlflow

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install Dependencies
        run: |
          pip install mlflow==2.19.0 dagshub pandas scikit-learn matplotlib seaborn

      - name: Train Model & Log to DagsHub
        working-directory: Workflow-CI/MLProject
        run: |
          python modelling.py

      - name: Get Latest Run ID
        id: get_run_id
        run: |
          # Ambil Run ID terakhir dari eksperimen "Water_Quality_Basic"
          RUN_ID=$(python -c "import mlflow; mlflow.set_tracking_uri('${{ env.MLFLOW_TRACKING_URI }}'); print(mlflow.search_runs(experiment_names=['Water_Quality_Basic'], order_by=['start_time DESC'], max_results=1).iloc[0].run_id)")
          echo "Run ID Found: $RUN_ID"
          echo "run_id=$RUN_ID" >> $GITHUB_OUTPUT

      - name: Build Docker Image using MLflow
        run: |
          IMAGE_TAG="${{ secrets.DOCKER_HUB_USERNAME }}/water-quality-model:latest"
          
          echo "Building Docker Image: $IMAGE_TAG from Run ID: ${{ steps.get_run_id.outputs.run_id }}"
          
          mlflow models build-docker \
            --model-uri "runs:/${{ steps.get_run_id.outputs.run_id }}/model" \
            --name "$IMAGE_TAG"

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}

      - name: Push Docker Image
        run: |
          docker push ${{ secrets.DOCKER_HUB_USERNAME }}/water-quality-model:latest