#!/bin/bash

# ScrambleBench Contamination Detection Pipeline
# Comprehensive contamination testing for gemma3:4b model

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="gemma3:4b"
CONFIG_FILE="configs/evaluation/contamination_detection_gemma3_4b.yaml"
OUTPUT_DIR="data/reports/contamination_detection_gemma3_4b"
BENCHMARK_FILE="data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json"

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                     🔬 ScrambleBench                          ║
║                Contamination Detection Pipeline               ║
║                                                               ║
║  Comprehensive analysis of training data contamination        ║
║  through performance comparison on scrambled questions        ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Print section header
print_section() {
    echo -e "\n${CYAN}$1${NC}"
    echo "$(printf '=%.0s' {1..60})"
}

# Print step
print_step() {
    echo -e "${BLUE}➤ $1${NC}"
}

# Print success
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print warning
print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Print error
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_section "🔍 Checking Prerequisites"
    
    # Check Python environment
    print_step "Checking Python environment..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check Ollama
    print_step "Checking Ollama installation..."
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama is required but not installed"
        echo "Install from: https://ollama.ai/"
        exit 1
    fi
    print_success "Ollama found: $(ollama --version)"
    
    # Check if Ollama service is running
    print_step "Checking Ollama service..."
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_error "Ollama service is not running"
        echo "Start with: ollama serve"
        exit 1
    fi
    print_success "Ollama service is running"
    
    # Check model availability
    print_step "Checking model availability: $MODEL_NAME"
    if ! ollama list | grep -q "$MODEL_NAME"; then
        print_warning "Model $MODEL_NAME not found"
        echo "Pulling model..."
        ollama pull "$MODEL_NAME"
        print_success "Model $MODEL_NAME pulled successfully"
    else
        print_success "Model $MODEL_NAME is available"
    fi
    
    # Check benchmark data
    print_step "Checking benchmark data..."
    if [[ ! -f "$BENCHMARK_FILE" ]]; then
        print_error "Benchmark file not found: $BENCHMARK_FILE"
        exit 1
    fi
    print_success "Benchmark data found"
    
    # Check Python dependencies
    print_step "Checking Python dependencies..."
    python3 -c "import numpy, pandas, matplotlib, seaborn, scipy, sklearn" 2>/dev/null || {
        print_warning "Some Python dependencies missing. Installing..."
        pip install numpy pandas matplotlib seaborn scipy scikit-learn
    }
    print_success "Python dependencies verified"
}

# Setup directories
setup_directories() {
    print_section "📁 Setting Up Directories"
    
    print_step "Creating output directories..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "data/languages"
    mkdir -p "logs"
    
    print_success "Directories created"
}

# Run contamination detection
run_contamination_detection() {
    print_section "🧪 Running Contamination Detection Analysis"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/contamination_detection_${timestamp}.log"
    
    print_step "Starting contamination analysis..."
    print_step "Log file: $log_file"
    print_step "Configuration: $CONFIG_FILE"
    
    # Run the main analysis script
    if python3 run_scrambled_comparison.py \
        --config "$CONFIG_FILE" \
        --verbose \
        --output-dir "$OUTPUT_DIR" 2>&1 | tee "$log_file"; then
        print_success "Contamination detection completed successfully"
    else
        print_error "Contamination detection failed"
        echo "Check log file: $log_file"
        exit 1
    fi
}

# Run advanced analysis
run_advanced_analysis() {
    print_section "📊 Running Advanced Analysis"
    
    print_step "Generating advanced contamination insights..."
    
    if python3 contamination_analyzer.py \
        "$OUTPUT_DIR/contamination_report.json" \
        --verbose; then
        print_success "Advanced analysis completed"
    else
        print_warning "Advanced analysis failed (optional step)"
    fi
}

# Generate comprehensive report
generate_report() {
    print_section "📄 Generating Comprehensive Report"
    
    local report_file="$OUTPUT_DIR/FINAL_CONTAMINATION_REPORT.md"
    
    print_step "Creating comprehensive report..."
    
    cat > "$report_file" << EOF
# ScrambleBench Contamination Detection Report

**Model:** $MODEL_NAME  
**Analysis Date:** $(date)  
**Analysis Type:** Comprehensive Contamination Detection  

## Executive Summary

This report presents the results of a comprehensive contamination detection analysis
for the $MODEL_NAME model using ScrambleBench's contamination-resistant evaluation
methodology.

## Methodology

ScrambleBench detects potential training data contamination by comparing model
performance on original questions versus semantically equivalent but surface-form
different "scrambled" versions of the same questions.

### Scrambling Techniques Applied:
1. **Constructed Language Translation** - Converting questions to artificial languages
2. **Synonym Replacement** - Replacing words with synonyms while preserving meaning
3. **Proper Noun Swapping** - Replacing proper nouns with thematically appropriate alternatives

### Contamination Detection Logic:
- **High Performance Drop** → Potential contamination (model memorized surface forms)
- **Stable Performance** → Clean training (model understands semantic content)

## Key Files Generated

EOF

    # Add file listings
    find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.md" -o -name "*.png" | sort | while read -r file; do
        echo "- \`$(basename "$file")\` - $(basename "$file" | sed 's/_/ /g' | sed 's/.json//g' | sed 's/.md//g' | sed 's/.png//g')" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

## How to Interpret Results

### Contamination Resistance Score
- **0.8 - 1.0**: Excellent resistance (low contamination risk)
- **0.6 - 0.8**: Good resistance (moderate contamination risk)  
- **0.0 - 0.6**: Poor resistance (high contamination risk)

### Performance Drop Analysis
- **< 5%**: Normal variation
- **5% - 15%**: Moderate concern
- **> 15%**: Significant concern (potential contamination)

### Statistical Significance
- **p < 0.05**: Statistically significant performance drop
- **p ≥ 0.05**: No significant evidence of contamination

## Next Steps

1. **If Low Risk**: Continue using model with confidence
2. **If Moderate Risk**: Investigate specific vulnerable transformations
3. **If High Risk**: Consider training data audit and model retraining

---

*Generated by ScrambleBench Contamination Detection Pipeline*
EOF

    print_success "Comprehensive report generated: $report_file"
}

# Display results
display_results() {
    print_section "📊 Analysis Results"
    
    # Check if contamination report exists
    if [[ -f "$OUTPUT_DIR/contamination_report.json" ]]; then
        print_step "Reading contamination analysis results..."
        
        # Extract key metrics using Python
        python3 << EOF
import json
import sys

try:
    with open('$OUTPUT_DIR/contamination_report.json', 'r') as f:
        report = json.load(f)
    
    model_name = report['model_name']
    total_questions = report['total_questions']
    overall_score = report['overall_contamination_score']
    high_risk_transforms = report['high_risk_transformations']
    
    print(f"Model: {model_name}")
    print(f"Questions Analyzed: {total_questions}")
    print(f"Overall Contamination Resistance Score: {overall_score:.3f}")
    
    if overall_score >= 0.8:
        print("Assessment: ✅ LOW contamination risk")
    elif overall_score >= 0.6:
        print("Assessment: ⚠️ MODERATE contamination risk")
    else:
        print("Assessment: 🚨 HIGH contamination risk")
    
    if high_risk_transforms:
        print(f"High-Risk Transformations: {', '.join(high_risk_transforms)}")
    else:
        print("High-Risk Transformations: None detected")
    
    print("\nTransformation Details:")
    for analysis in report['transformation_analyses']:
        t_type = analysis['transformation_type']
        orig_acc = analysis['original_accuracy']
        scram_acc = analysis['scrambled_accuracy']
        drop_pct = analysis['performance_drop_percent']
        resistance = analysis['contamination_resistance_score']
        
        print(f"  {t_type}:")
        print(f"    Original Accuracy: {orig_acc:.3f}")
        print(f"    Scrambled Accuracy: {scram_acc:.3f}")
        print(f"    Performance Drop: {drop_pct:.1f}%")
        print(f"    Resistance Score: {resistance:.3f}")

except Exception as e:
    print(f"Error reading results: {e}")
    sys.exit(1)
EOF
        
    else
        print_warning "Contamination report not found"
    fi
    
    # List generated files
    print_step "Generated files:"
    if [[ -d "$OUTPUT_DIR" ]]; then
        find "$OUTPUT_DIR" -type f | sort | while read -r file; do
            echo "  📄 $file"
        done
    fi
}

# Cleanup function
cleanup() {
    print_section "🧹 Cleanup"
    
    # Optional: Clean up temporary files
    # This is kept minimal to preserve results
    print_step "Cleaning up temporary files..."
    
    # Remove any temporary language files older than 1 day
    find data/languages -name "*.json" -mtime +1 -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main execution
main() {
    print_banner
    
    # Parse command line arguments
    QUICK_MODE=false
    SKIP_CHECKS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --quick        Run in quick mode (fewer samples)"
                echo "  --skip-checks  Skip prerequisite checks"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Modify for quick mode
    if [[ "$QUICK_MODE" == true ]]; then
        print_warning "Running in QUICK MODE (reduced samples for testing)"
        QUICK_FLAG="--quick"
    else
        QUICK_FLAG=""
    fi
    
    # Execute pipeline
    if [[ "$SKIP_CHECKS" != true ]]; then
        check_prerequisites
    fi
    
    setup_directories
    
    # Modify the contamination detection call to include quick flag if set
    if [[ "$QUICK_MODE" == true ]]; then
        print_section "🧪 Running Contamination Detection Analysis (Quick Mode)"
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local log_file="logs/contamination_detection_${timestamp}.log"
        
        print_step "Starting contamination analysis (quick mode)..."
        if python3 run_scrambled_comparison.py \
            --config "$CONFIG_FILE" \
            --quick \
            --verbose \
            --output-dir "$OUTPUT_DIR" 2>&1 | tee "$log_file"; then
            print_success "Contamination detection completed successfully"
        else
            print_error "Contamination detection failed"
            exit 1
        fi
    else
        run_contamination_detection
    fi
    
    run_advanced_analysis
    generate_report
    display_results
    cleanup
    
    print_section "🎉 Pipeline Complete"
    print_success "ScrambleBench contamination detection analysis completed!"
    print_step "Results directory: $OUTPUT_DIR"
    print_step "Main report: $OUTPUT_DIR/CONTAMINATION_SUMMARY.md"
    print_step "Detailed data: $OUTPUT_DIR/contamination_report.json"
    
    echo -e "\n${GREEN}┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│  🔬 ScrambleBench Contamination Detection Complete  │${NC}"
    echo -e "${GREEN}│                                                     │${NC}"
    echo -e "${GREEN}│  Check the results in:                              │${NC}"
    echo -e "${GREEN}│  📁 $OUTPUT_DIR/               │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────────┘${NC}"
}

# Handle script interruption
trap 'print_error "Script interrupted"; exit 130' INT TERM

# Execute main function
main "$@"