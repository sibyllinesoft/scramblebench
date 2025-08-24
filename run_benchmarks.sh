#!/bin/bash
"""
Quick Benchmark Runner for Ollama Models
========================================

This script provides a convenient way to run benchmarks on the target models
with appropriate error handling and progress indicators.
"""

set -e  # Exit on any error

echo "🎯 Custom Ollama Benchmark Runner"
echo "================================="

# Check if Python environment is ready
if ! python3 -c "import scramblebench" 2>/dev/null; then
    echo "❌ ScrambleBench not found in Python path"
    echo "💡 Make sure you're in the virtual environment:"
    echo "   source venv/bin/activate"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama server not running"
    echo "💡 Start Ollama server:"
    echo "   ollama serve"
    exit 1
fi

echo "✅ Environment checks passed"
echo ""

# Function to run benchmark for a model
run_benchmark() {
    local model=$1
    local samples=${2:-15}  # Default to 15 questions
    
    echo "🚀 Running benchmark for: $model"
    echo "📊 Questions: $samples"
    echo "⏱️  Starting..."
    echo ""
    
    if python3 run_custom_ollama_benchmarks.py --model "$model" --samples "$samples" --verbose; then
        echo ""
        echo "✅ Benchmark completed for $model"
        echo "📁 Results saved in data/results/custom_benchmarks/${model//:/_}/"
        echo ""
    else
        echo ""
        echo "❌ Benchmark failed for $model"
        echo "💡 Check the logs above for details"
        echo ""
        return 1
    fi
}

# Function to check if model is available
check_model() {
    local model=$1
    echo "🔍 Checking model availability: $model"
    
    if python3 run_custom_ollama_benchmarks.py --list-models | grep -q "$model"; then
        echo "✅ Model $model is available"
        return 0
    else
        echo "❌ Model $model not found"
        echo "💡 Pull the model:"
        echo "   ollama pull $model"
        return 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    "gemma3" | "gemma3:4b")
        if check_model "gemma3:4b"; then
            run_benchmark "gemma3:4b" "${2:-15}"
        fi
        ;;
    "gpt-oss" | "gpt-oss:20b")
        if check_model "gpt-oss:20b"; then
            run_benchmark "gpt-oss:20b" "${2:-15}"
        fi
        ;;
    "both" | "all")
        echo "🎯 Running benchmarks for both target models"
        echo ""
        
        if check_model "gemma3:4b"; then
            run_benchmark "gemma3:4b" "${2:-15}"
            echo "⏳ Brief pause before next model..."
            sleep 3
        fi
        
        if check_model "gpt-oss:20b"; then
            run_benchmark "gpt-oss:20b" "${2:-15}"
        fi
        
        echo "🎉 All benchmarks completed!"
        echo "📊 Compare results in data/results/custom_benchmarks/"
        ;;
    "list" | "--list")
        echo "📋 Available Models:"
        python3 run_custom_ollama_benchmarks.py --list-models
        ;;
    "help" | "--help" | "-h" | "")
        echo "Usage: $0 <model> [num_questions]"
        echo ""
        echo "Models:"
        echo "  gemma3      - Run benchmark on gemma3:4b"
        echo "  gpt-oss     - Run benchmark on gpt-oss:20b"
        echo "  both        - Run benchmarks on both models"
        echo "  list        - List available models"
        echo "  help        - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 gemma3           # Run gemma3:4b with 15 questions"
        echo "  $0 gpt-oss 20       # Run gpt-oss:20b with 20 questions"
        echo "  $0 both 25          # Run both models with 25 questions each"
        echo "  $0 list             # Show available models"
        echo ""
        echo "Prerequisites:"
        echo "  1. Ollama server running: ollama serve"
        echo "  2. Models pulled: ollama pull <model>"
        echo "  3. ScrambleBench environment active"
        ;;
    *)
        echo "❌ Unknown model: $1"
        echo "💡 Use '$0 help' for usage information"
        exit 1
        ;;
esac