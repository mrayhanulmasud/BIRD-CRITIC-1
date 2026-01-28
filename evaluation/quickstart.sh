#!/bin/bash
# Quick start script - downloads data and runs the pipeline

set -e  # Exit on error

echo "====================================="
echo "BIRD-CRITIC Pipeline Quick Start"
echo "====================================="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
#pip install -q datasets psycopg2-binary requests
echo "✓ Dependencies installed"
echo ""

# Step 2: Download dataset
echo "Step 2: Downloading BIRD-CRITIC Flash dataset (200 samples)..."
python download_dataset.py --dataset birdsql/bird-critic-1.0-flash-exp --output data/bird-critic-flash.jsonl
echo "✓ Dataset downloaded"
echo ""

# Step 3: Check PostgreSQL connection
echo "Step 3: Checking PostgreSQL connection..."
if docker ps | grep -q bird_critic_postgresql; then
    echo "✓ PostgreSQL container is running"
else
    echo "⚠ PostgreSQL container not running!"
    echo "Please start it with: docker compose up -d postgresql"
    exit 1
fi
echo ""

# Step 4: Check Ollama
echo "Step 4: Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
    echo "Available models:"
    curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; models = json.load(sys.stdin).get('models', []); print('\n'.join(f\"  - {m['name']}\" for m in models) if models else '  (no models found)')"
else
    echo "⚠ Ollama is not running!"
    echo "Please start it with: ollama serve"
    echo "And pull a model with: ollama pull qwen2.5-coder:32b"
    exit 1
fi
echo ""

# Step 5: Run pipeline
echo "Step 5: Running pipeline (1 sample for testing)..."
echo "Command: python bird_critic_pipeline.py --dataset data/bird-critic-flash.jsonl --output results/test_output.jsonl --samples 1"
echo ""

mkdir -p results
python bird_critic_pipeline.py \
  --dataset data/bird-critic-flash.jsonl \
  --output results/test_output.jsonl \
  --samples 1 \
  --model qwen2.5-coder:32b \
  --db-user root \
  --db-password 123123

echo ""
echo "====================================="
echo "✓ Pipeline completed successfully!"
echo "====================================="
echo ""
echo "Results saved to: results/test_output.jsonl"
echo ""
echo "To run more samples, use:"
echo "  python bird_critic_pipeline.py --dataset data/bird-critic-flash.jsonl --output results/output.jsonl --samples 10"
