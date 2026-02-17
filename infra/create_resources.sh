#!/bin/bash
# Create GCP resources for RAG Copilot
# Usage: ./create_resources.sh

set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID is required}"
: "${GCP_REGION:=us-central1}"
: "${BQ_DATASET:=rag_copilot}"
: "${GCS_BUCKET:?GCS_BUCKET is required}"

echo "=== Creating GCP Resources for RAG Copilot ==="
echo "Project: $GCP_PROJECT_ID"
echo "Region: $GCP_REGION"
echo "Dataset: $BQ_DATASET"
echo "Bucket: $GCS_BUCKET"
echo ""

# Set project
gcloud config set project "$GCP_PROJECT_ID"

# Enable required APIs
echo "Enabling APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

# Create GCS bucket
echo "Creating GCS bucket..."
gcloud storage buckets create "gs://$GCS_BUCKET" \
    --location="$GCP_REGION" \
    --uniform-bucket-level-access \
    2>/dev/null || echo "Bucket already exists or error occurred"

# Create BigQuery dataset
echo "Creating BigQuery dataset..."
bq --location=US mk --dataset \
    --description "RAG Copilot document chunks" \
    "$GCP_PROJECT_ID:$BQ_DATASET" \
    2>/dev/null || echo "Dataset already exists or error occurred"

# Create service accounts
echo "Creating service accounts..."

# API service account
gcloud iam service-accounts create rag-api \
    --display-name="RAG Copilot API" \
    2>/dev/null || echo "Service account rag-api already exists"

# Ingestion service account
gcloud iam service-accounts create rag-ingest \
    --display-name="RAG Copilot Ingestion" \
    2>/dev/null || echo "Service account rag-ingest already exists"

# Assign roles to API service account
echo "Assigning roles to rag-api..."
API_SA="rag-api@$GCP_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$API_SA" \
    --role="roles/bigquery.dataViewer" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$API_SA" \
    --role="roles/bigquery.jobUser" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$API_SA" \
    --role="roles/aiplatform.user" \
    --quiet

# Assign roles to Ingestion service account
echo "Assigning roles to rag-ingest..."
INGEST_SA="rag-ingest@$GCP_PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/bigquery.dataEditor" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/bigquery.jobUser" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/storage.objectViewer" \
    --quiet

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
    --member="serviceAccount:$INGEST_SA" \
    --role="roles/aiplatform.user" \
    --quiet

# Create Artifact Registry repository
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create rag-copilot \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="RAG Copilot container images" \
    2>/dev/null || echo "Repository already exists"

echo ""
echo "=== Resource Creation Complete ==="
echo ""
echo "Service Accounts:"
echo "  API: $API_SA"
echo "  Ingestion: $INGEST_SA"
echo ""
echo "Next steps:"
echo "  1. Upload documents to gs://$GCS_BUCKET/documents/"
echo "  2. Run ./infra/run_ingest.sh to ingest documents"
echo "  3. Run ./infra/deploy_cloudrun.sh to deploy the API"
