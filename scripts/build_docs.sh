#!/bin/bash
"""
Documentation build script for ScrambleBench.

This script builds the Sphinx documentation with various options
and handles common documentation build tasks.
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="html"
CLEAN=false
VERBOSE=false
OPEN_BROWSER=false
CHECK_LINKS=false
AUTO_RELOAD=false
OUTPUT_DIR=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_help() {
    cat << EOF
Documentation Build Script for ScrambleBench

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Build type: html, latex, pdf, epub (default: html)
    -c, --clean             Clean build directory before building
    -v, --verbose           Verbose output
    -o, --open              Open documentation in browser after building
    -l, --check-links       Check for broken links
    -w, --watch             Auto-reload documentation on changes
    -d, --output-dir DIR    Custom output directory

EXAMPLES:
    $0                      # Build HTML documentation
    $0 -c -v                # Clean build with verbose output
    $0 -t pdf               # Build PDF documentation
    $0 -w                   # Auto-reload development server
    $0 -l                   # Check for broken links
    $0 -o                   # Build and open in browser

REQUIREMENTS:
    - Sphinx and dependencies installed (uv sync --group docs)
    - For PDF: LaTeX distribution (texlive-latex-recommended)
    - For auto-reload: sphinx-autobuild (pip install sphinx-autobuild)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -o|--open)
            OPEN_BROWSER=true
            shift
            ;;
        -l|--check-links)
            CHECK_LINKS=true
            shift
            ;;
        -w|--watch)
            AUTO_RELOAD=true
            shift
            ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "$DOCS_DIR/conf.py" ]]; then
    print_error "Sphinx configuration not found. Run from project root or ensure docs/conf.py exists."
    exit 1
fi

# Check if documentation dependencies are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v sphinx-build &> /dev/null; then
        print_error "sphinx-build not found. Install with: uv sync --group docs"
        exit 1
    fi
    
    if [[ "$AUTO_RELOAD" == true ]] && ! command -v sphinx-autobuild &> /dev/null; then
        print_error "sphinx-autobuild not found. Install with: pip install sphinx-autobuild"
        exit 1
    fi
    
    if [[ "$BUILD_TYPE" == "pdf" ]] && ! command -v pdflatex &> /dev/null; then
        print_warning "pdflatex not found. PDF generation may fail."
        print_warning "Install LaTeX: sudo apt-get install texlive-latex-recommended texlive-fonts-recommended"
    fi
    
    print_success "Dependencies check passed"
}

# Set output directory based on build type
set_output_dir() {
    if [[ -n "$OUTPUT_DIR" ]]; then
        BUILD_DIR="$OUTPUT_DIR"
    else
        BUILD_DIR="$DOCS_DIR/_build/$BUILD_TYPE"
    fi
}

# Clean build directory
clean_build() {
    if [[ "$CLEAN" == true ]]; then
        print_status "Cleaning build directory..."
        rm -rf "$DOCS_DIR/_build"
        print_success "Build directory cleaned"
    fi
}

# Build documentation
build_docs() {
    print_status "Building $BUILD_TYPE documentation..."
    
    cd "$DOCS_DIR"
    
    # Set Sphinx options
    SPHINX_OPTS=""
    if [[ "$VERBOSE" == true ]]; then
        SPHINX_OPTS="$SPHINX_OPTS -v"
    fi
    
    case $BUILD_TYPE in
        html)
            sphinx-build -b html $SPHINX_OPTS . "$BUILD_DIR"
            ;;
        latex)
            sphinx-build -b latex $SPHINX_OPTS . "$BUILD_DIR"
            ;;
        pdf)
            # Build LaTeX first, then compile to PDF
            sphinx-build -b latex $SPHINX_OPTS . "$DOCS_DIR/_build/latex"
            cd "$DOCS_DIR/_build/latex"
            make all-pdf
            # Copy PDF to output directory if different
            if [[ "$BUILD_DIR" != "$DOCS_DIR/_build/latex" ]]; then
                mkdir -p "$BUILD_DIR"
                cp *.pdf "$BUILD_DIR/"
            fi
            cd "$DOCS_DIR"
            ;;
        epub)
            sphinx-build -b epub $SPHINX_OPTS . "$BUILD_DIR"
            ;;
        *)
            print_error "Unknown build type: $BUILD_TYPE"
            exit 1
            ;;
    esac
    
    if [[ $? -eq 0 ]]; then
        print_success "$BUILD_TYPE documentation built successfully"
        print_status "Output directory: $BUILD_DIR"
    else
        print_error "Documentation build failed"
        exit 1
    fi
}

# Check for broken links
check_broken_links() {
    if [[ "$CHECK_LINKS" == true ]]; then
        print_status "Checking for broken links..."
        cd "$DOCS_DIR"
        sphinx-build -b linkcheck . "_build/linkcheck"
        
        if [[ -f "_build/linkcheck/output.txt" ]]; then
            broken_links=$(grep -c "broken" "_build/linkcheck/output.txt" || true)
            if [[ $broken_links -gt 0 ]]; then
                print_warning "Found $broken_links broken links"
                print_status "Check _build/linkcheck/output.txt for details"
            else
                print_success "No broken links found"
            fi
        fi
    fi
}

# Open documentation in browser
open_browser() {
    if [[ "$OPEN_BROWSER" == true ]] && [[ "$BUILD_TYPE" == "html" ]]; then
        print_status "Opening documentation in browser..."
        
        # Find the main HTML file
        if [[ -f "$BUILD_DIR/index.html" ]]; then
            # Try different browser commands
            if command -v xdg-open &> /dev/null; then
                xdg-open "$BUILD_DIR/index.html"
            elif command -v open &> /dev/null; then
                open "$BUILD_DIR/index.html"
            elif command -v start &> /dev/null; then
                start "$BUILD_DIR/index.html"
            else
                print_warning "Could not detect browser command"
                print_status "Open manually: file://$BUILD_DIR/index.html"
            fi
        else
            print_warning "index.html not found in build directory"
        fi
    fi
}

# Auto-reload development server
auto_reload_server() {
    if [[ "$AUTO_RELOAD" == true ]]; then
        print_status "Starting auto-reload development server..."
        print_status "Documentation will be available at http://localhost:8000"
        print_status "Press Ctrl+C to stop"
        
        cd "$DOCS_DIR"
        sphinx-autobuild . "_build/html" --host 0.0.0.0 --port 8000 $SPHINX_OPTS
    fi
}

# Generate API documentation
generate_api_docs() {
    print_status "Generating API documentation..."
    
    cd "$PROJECT_ROOT"
    
    # Use sphinx-apidoc to generate API documentation
    if command -v sphinx-apidoc &> /dev/null; then
        sphinx-apidoc -f -o "$DOCS_DIR/api_generated" src/scramblebench
        print_success "API documentation generated"
    else
        print_warning "sphinx-apidoc not found, skipping API generation"
    fi
}

# Main execution
main() {
    print_status "ScrambleBench Documentation Builder"
    print_status "====================================="
    
    check_dependencies
    set_output_dir
    clean_build
    
    # Generate API docs if building HTML
    if [[ "$BUILD_TYPE" == "html" ]]; then
        generate_api_docs
    fi
    
    if [[ "$AUTO_RELOAD" == true ]]; then
        auto_reload_server
    else
        build_docs
        check_broken_links
        open_browser
        
        print_success "Documentation build complete!"
        
        case $BUILD_TYPE in
            html)
                print_status "View at: file://$BUILD_DIR/index.html"
                ;;
            pdf)
                print_status "PDF files in: $BUILD_DIR"
                ;;
            *)
                print_status "Output in: $BUILD_DIR"
                ;;
        esac
    fi
}

# Run main function
main "$@"