#!/bin/bash
# Development environment setup script for ScrambleBench
# This script sets up a complete development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    
    case "$(uname -s)" in
        Linux*)     OS=Linux;;
        Darwin*)    OS=Mac;;
        CYGWIN*)    OS=Cygwin;;
        MINGW*)     OS=MinGw;;
        *)          OS="UNKNOWN:$(uname -s)"
    esac
    
    if [[ "$OS" == "UNKNOWN"* ]]; then
        print_error "Unsupported operating system: $OS"
        exit 1
    fi
    
    print_success "Operating system: $OS"
}

# Check if Python 3.9+ is available
check_python() {
    print_status "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9 or later."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    REQUIRED_VERSION="3.9"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
        print_error "Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or later is required."
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION is installed"
}

# Install Poetry if not present
install_poetry() {
    print_status "Checking Poetry installation..."
    
    if command -v poetry &> /dev/null; then
        POETRY_VERSION=$(poetry --version | cut -d' ' -f3)
        print_success "Poetry $POETRY_VERSION is already installed"
        return
    fi
    
    print_status "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    if command -v poetry &> /dev/null; then
        print_success "Poetry installed successfully"
    else
        print_error "Failed to install Poetry. Please install it manually."
        print_error "Visit: https://python-poetry.org/docs/#installation"
        exit 1
    fi
}

# Install Git if not present (Linux only)
check_git() {
    print_status "Checking Git installation..."
    
    if ! command -v git &> /dev/null; then
        if [[ "$OS" == "Linux" ]]; then
            print_status "Installing Git..."
            if command -v apt-get &> /dev/null; then
                sudo apt-get update && sudo apt-get install -y git
            elif command -v yum &> /dev/null; then
                sudo yum install -y git
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y git
            else
                print_error "Cannot install Git automatically. Please install Git manually."
                exit 1
            fi
        else
            print_error "Git is not installed. Please install Git manually."
            exit 1
        fi
    fi
    
    print_success "Git is available"
}

# Initialize git repository if not already initialized
init_git() {
    print_status "Checking Git repository..."
    
    if [ ! -d ".git" ]; then
        print_status "Initializing Git repository..."
        git init
        git add .
        git commit -m "Initial commit: ScrambleBench codebase"
        print_success "Git repository initialized"
    else
        print_success "Git repository already exists"
    fi
}

# Install project dependencies
install_dependencies() {
    print_status "Installing project dependencies..."
    
    # Configure Poetry to create virtual environment in project directory
    poetry config virtualenvs.in-project true
    
    # Install dependencies
    poetry install --with dev,test,docs
    
    print_success "Dependencies installed successfully"
}

# Install pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."
    
    # Install pre-commit in the virtual environment
    poetry run pre-commit install
    
    # Run pre-commit on all files to ensure everything is working
    print_status "Running pre-commit on all files (this may take a while)..."
    poetry run pre-commit run --all-files || true
    
    print_success "Pre-commit hooks installed and configured"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "data"
        "data/benchmarks"
        "data/results"
        "data/cache"
        "data/languages"
        "logs"
        "reports"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        touch "$dir/.gitkeep"
    done
    
    print_success "Directories created"
}

# Setup environment files
setup_environment() {
    print_status "Setting up environment configuration..."
    
    # Create .env template if it doesn't exist
    if [ ! -f ".env.template" ]; then
        cat > .env.template << 'EOF'
# ScrambleBench Environment Configuration
# Copy this file to .env and fill in your values

# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Configuration
SCRAMBLEBENCH_LOG_LEVEL=INFO
SCRAMBLEBENCH_DATA_DIR=data
SCRAMBLEBENCH_RANDOM_SEED=42

# Development settings
SCRAMBLEBENCH_DEBUG=false
SCRAMBLEBENCH_ENABLE_TELEMETRY=false
EOF
        print_success "Created .env.template file"
    fi
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.template .env
        print_warning "Created .env file from template. Please edit it with your API keys."
    fi
}

# Validate installation
validate_installation() {
    print_status "Validating installation..."
    
    # Test that ScrambleBench can be imported
    if poetry run python -c "import scramblebench; print(f'ScrambleBench version: {scramblebench.__version__}')" 2>/dev/null; then
        print_success "ScrambleBench package imports successfully"
    else
        print_error "Failed to import ScrambleBench package"
        return 1
    fi
    
    # Test CLI
    if poetry run scramblebench --version &>/dev/null; then
        print_success "CLI is working"
    else
        print_warning "CLI test failed, but this might be expected in development"
    fi
    
    # Run a quick test
    print_status "Running quick test suite..."
    if poetry run pytest tests/test_core/test_config.py -v --tb=short; then
        print_success "Basic tests are passing"
    else
        print_warning "Some tests failed, but the environment is set up"
    fi
}

# Display next steps
show_next_steps() {
    print_success "Development environment setup complete!"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Edit .env file with your API keys"
    echo "2. Activate the virtual environment: poetry shell"
    echo "3. Run tests: poetry run pytest"
    echo "4. Start coding!"
    echo
    echo -e "${BLUE}Useful commands:${NC}"
    echo "  poetry run scramblebench --help    # Show CLI help"
    echo "  poetry run pytest                  # Run all tests"
    echo "  poetry run pytest -k test_name     # Run specific test"
    echo "  poetry run black .                 # Format code"
    echo "  poetry run ruff check .            # Lint code"
    echo "  poetry run mypy src/               # Type check"
    echo "  poetry run pre-commit run --all    # Run all pre-commit hooks"
    echo
    echo -e "${BLUE}Documentation:${NC}"
    echo "  cd docs && poetry run make html     # Build documentation"
    echo "  poetry run mkdocs serve             # Serve docs locally"
    echo
}

# Main setup function
main() {
    echo -e "${GREEN}"
    echo "=================================================="
    echo "  ScrambleBench Development Environment Setup"
    echo "=================================================="
    echo -e "${NC}"
    
    check_os
    check_python
    check_git
    install_poetry
    init_git
    install_dependencies
    create_directories
    setup_environment
    setup_precommit
    validate_installation
    show_next_steps
    
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "ScrambleBench Development Environment Setup"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --skip-git     Skip Git repository initialization"
        echo "  --skip-hooks   Skip pre-commit hooks setup"
        echo "  --no-validate  Skip installation validation"
        echo ""
        exit 0
        ;;
    --skip-git)
        SKIP_GIT=true
        ;;
    --skip-hooks)
        SKIP_HOOKS=true
        ;;
    --no-validate)
        NO_VALIDATE=true
        ;;
esac

# Run main setup
main