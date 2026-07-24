#!/bin/bash

# Postman API Benchmarking Script
# Run comprehensive API benchmarks using Newman (Postman CLI)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 OctoTetrahedral AGI - Postman Benchmark Suite${NC}"
echo "=================================================="
echo ""

# Check if Newman is installed
if ! command -v newman &> /dev/null; then
    echo -e "${YELLOW}⚠️  Newman not found. Installing...${NC}"
    npm install -g newman
fi

# Create results directory
mkdir -p benchmark_results/postman

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="benchmark_results/postman/benchmark_${TIMESTAMP}.json"

echo -e "${BLUE}📊 Running benchmarks...${NC}"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Run Newman with collection and environment
newman run postman_collection.json \
    -e postman_environment.json \
    --reporters cli,json,html \
    --reporter-json-export "$RESULTS_FILE" \
    --reporter-html-export "benchmark_results/postman/benchmark_${TIMESTAMP}.html" \
    --delay-request 500 \
    --timeout 60000

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Benchmarks completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📈 Results:${NC}"
    echo "  JSON: $RESULTS_FILE"
    echo "  HTML: benchmark_results/postman/benchmark_${TIMESTAMP}.html"
    echo ""
    
    # Parse and display summary
    echo -e "${BLUE}📊 Summary:${NC}"
    if command -v jq &> /dev/null; then
        STATS=$(jq '.run | {total: .stats.requests.total, success: .stats.requests.success, failed: .stats.requests.failed, duration: .timings.completed}' "$RESULTS_FILE")
        echo "$STATS" | jq '.'
    else
        echo "  Install jq to see detailed statistics"
    fi
else
    echo -e "${RED}❌ Benchmarks failed!${NC}"
    exit 1
fi
