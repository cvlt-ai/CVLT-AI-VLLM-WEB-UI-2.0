#!/bin/bash
#
# CVLT AI vLLM Gradio WebUI 2.0 - Startup Script
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "============================================="
    echo "   CVLT AI vLLM Gradio WebUI 2.0"
    echo "============================================="
    echo -e "${NC}"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
        python3 -m venv venv
        echo -e "${GREEN}Virtual environment created.${NC}"
    fi
}

# Activate virtual environment
activate_venv() {
    source venv/bin/activate
}

# Install dependencies
install_deps() {
    echo -e "${YELLOW}Installing/updating dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed.${NC}"
}

# Check for GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            echo -e "${GREEN}✓ GPU detected: $GPU_COUNT GPU(s)${NC}"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        else
            echo -e "${YELLOW}⚠ No GPU detected. Running in CPU mode (slow).${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ nvidia-smi not found. Running in CPU mode.${NC}"
    fi
}

# Load environment variables
load_env() {
    if [ -f ".env" ]; then
        echo -e "${GREEN}Loading environment from .env${NC}"
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo -e "${YELLOW}No .env file found. Using defaults.${NC}"
        echo -e "${YELLOW}Copy .env.example to .env to customize settings.${NC}"
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --setup      Install dependencies and setup environment"
    echo "  --gpu        Show GPU information and exit"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --setup    # Install dependencies"
    echo "  $0            # Start the application"
    echo ""
}

# Parse arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    print_banner
    show_help
    exit 0
fi

# Main execution
print_banner

# Check for --setup flag
if [ "$1" == "--setup" ]; then
    echo -e "${BLUE}Running setup...${NC}"
    check_venv
    activate_venv
    install_deps
    echo -e "${GREEN}Setup complete! Run '$0' to start the application.${NC}"
    exit 0
fi

# Check for --gpu flag
if [ "$1" == "--gpu" ]; then
    check_gpu
    exit 0
fi

# Normal startup
check_venv
activate_venv

# Check if dependencies are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo -e "${YELLOW}Dependencies not installed. Running setup...${NC}"
    install_deps
fi

load_env
check_gpu

echo -e "${GREEN}Starting vLLM Gradio WebUI...${NC}"
echo -e "${BLUE}Access the interface at: http://localhost:$PORT${NC}"
echo ""

# Run the application
exec python main.py
