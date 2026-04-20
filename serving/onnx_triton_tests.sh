#!/bin/bash

# ==========================================
# 1. PREPARATION
# ==========================================
echo "--- Generating input.json ---"
mkdir -p ~/triton_workspace/onnx_model
cat << 'EOF' > ~/triton_workspace/onnx_model/input.json
{
  "data": [
    {
      "INPUT_TEXT": ["Alice: We need to finish the UI this week. Bob: I will handle the backend API. Carol: Let's meet again on Friday."]
    }
  ]
}
EOF

# Stop any existing server just in case
echo "--- Cleaning up old containers ---"
docker stop triton_server 2>/dev/null

# ==========================================
# 2. LAUNCH TRITON SERVER
# ==========================================
echo "--- Launching Triton Server (CUDA 11 compatible) ---"
docker run --gpus=all --rm -d \
  --name triton_server \
  --shm-size=4g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  -v $(pwd)/onnx_model:/workspace/onnx_model \
  -e HF_HOME=/workspace/cache \
  nvcr.io/nvidia/tritonserver:22.12-py3 \
  bash -c "pip install optimum transformers onnxruntime-gpu==1.17.1 onnx tritonclient[all]==2.29.0 && tritonserver --model-repository=/models"

echo "Server is booting and installing dependencies in the background."
echo "Please wait approximately 60-90 seconds for the model to load..."

# Pause to allow the server to fully start and display "READY"
# (If running this manually, you can cancel the sleep and just check docker logs triton_server -f)
sleep 90 

# ==========================================
# 3. RUN BENCHMARKS
# ==========================================
echo "--- Running Baseline Benchmark (Concurrency 1) ---"
docker exec -it triton_server perf_analyzer \
  -u localhost:8000 \
  -m bart_onnx \
  --input-data /workspace/onnx_model/input.json \
  -b 1 \
  --shape INPUT_TEXT:1

echo "--- Running Heavy Load Benchmark (Concurrency 15) ---"
# Note: This will take a few minutes due to the 30-second measurement interval
docker exec -it triton_server perf_analyzer \
  -u localhost:8000 \
  -m bart_onnx \
  --input-data /workspace/onnx_model/input.json \
  -b 1 \
  --concurrency-range 15 \
  --shape INPUT_TEXT:1 \
  --measurement-interval 30000

echo "--- Benchmarks Complete! ---"