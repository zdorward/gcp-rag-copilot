#!/bin/bash
# Build and deploy to Cloud Run
# Usage: ./deploy_cloudrun.sh

set -euo pipefail

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
: "${EMBEDDING_MODEL:=text-embedding-004}"
: "${LLM_MODEL:=gemini-2.0-flash}"

SERVICE_NAME="rag-copilot-api"
IMAGE_NAME="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/rag-copilot/api"
API_SA="rag-api@$GCP_PROJECT_ID.iam.gserviceaccount.com"

echo "=== Deploying RAG Copilot to Cloud Run ==="
echo "Project: $GCP_PROJECT_ID"
echo "Region: $GCP_REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Set project
gcloud config set project "$GCP_PROJECT_ID"

# Configure Docker for Artifact Registry
echo "Configuring Docker..."
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev" --quiet

# Build the image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:latest" -f docker/Dockerfile .

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push "$IMAGE_NAME:latest"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE_NAME:latest" \
    --platform=managed \
    --region="$GCP_REGION" \
    --service-account="$API_SA" \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --timeout=300 \
    --concurrency=80 \
    --min-instances=0 \
    --max-instances=10 \
    --set-env-vars="GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_REGION=$GCP_REGION,BQ_DATASET=$BQ_DATASET,BQ_TABLE=$BQ_TABLE,GCS_BUCKET=$GCS_BUCKET,EMBEDDING_MODEL=$EMBEDDING_MODEL,LLM_MODEL=$LLM_MODEL,LOG_LEVEL=INFO"

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --platform=managed \
    --region="$GCP_REGION" \
    --format="value(status.url)")

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the API:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "  curl -X POST $SERVICE_URL/ask \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"What is in the documents?\", \"top_k\": 5}'"
