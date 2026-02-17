#!/bin/bash
# Run document ingestion
# Usage: ./run_ingest.sh [local|cloud]

set -euo pipefail

MODE="${1:-local}"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${GCP_REGION:=us-central1}"
: "${BQ_DATASET:=rag_copilot}"
: "${BQ_TABLE:=chunks}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"
: "${GCS_PREFIX:=documents/}"
: "${EMBEDDING_MODEL:=text-embedding-004}"
: "${LLM_MODEL:=gemini-1.5-flash}"

echo "=== Running Document Ingestion ==="
echo "Mode: $MODE"
echo "Bucket: gs://$GCS_BUCKET/$GCS_PREFIX"
echo ""

if [ "$MODE" == "local" ]; then
    # Run locally (requires gcloud auth)
    echo "Running ingestion locally..."

    # Ensure dependencies are installed
    pip install -r requirements.txt --quiet

    # Run the ingestion script
    python -m ingestion.ingest

elif [ "$MODE" == "cloud" ]; then
    # Run as Cloud Run Job
    echo "Running ingestion as Cloud Run Job..."

    JOB_NAME="rag-copilot-ingest"
    IMAGE_NAME="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/rag-copilot/api"
    INGEST_SA="rag-ingest@$GCP_PROJECT_ID.iam.gserviceaccount.com"

    # Set project
    gcloud config set project "$GCP_PROJECT_ID"

    # Create or update Cloud Run Job
    gcloud run jobs deploy "$JOB_NAME" \
        --image="$IMAGE_NAME:latest" \
        --region="$GCP_REGION" \
        --service-account="$INGEST_SA" \
        --memory=2Gi \
        --cpu=2 \
        --task-timeout=3600 \
        --max-retries=1 \
        --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_REGION=$GCP_REGION,BQ_DATASET=$BQ_DATASET,BQ_TABLE=$BQ_TABLE,GCS_BUCKET=$GCS_BUCKET,GCS_PREFIX=$GCS_PREFIX,EMBEDDING_MODEL=$EMBEDDING_MODEL,LLM_MODEL=$LLM_MODEL,LOG_LEVEL=INFO" \
        --command="python,-m,ingestion.ingest"

    # Execute the job
    echo "Executing Cloud Run Job..."
    gcloud run jobs execute "$JOB_NAME" \
        --region="$GCP_REGION" \
        --wait

else
    echo "Usage: ./run_ingest.sh [local|cloud]"
    exit 1
fi

echo ""
echo "=== Ingestion Complete ==="
