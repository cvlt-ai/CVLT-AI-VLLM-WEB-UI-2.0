# CVLT AI vLLM Gradio WebUI 2.0

A comprehensive web-based interface for vLLM with advanced features including multi-GPU support, RAG capabilities, web scraping, file processing, and more.

##  Features

### Core Features
- **Advanced vLLM Integration**: Full support for vLLM nightly (experimental) version with cutting-edge features
- **Quantization Support**: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8 quantization for model compression
- **Multi-GPU Support**: Tensor parallelism and pipeline parallelism for distributed inference
- **Interactive Chat Interface**: Clean monochromatic Gradio-based UI with real-time streaming
- **Model Management**: Easy model discovery, loading, and switching

### RAG (Retrieval-Augmented Generation)
- **Vector Database**: ChromaDB integration for efficient document storage and retrieval
- **Multiple Document Formats**: Support for PDF, Word, text, images, and more
- **Semantic Search**: Advanced retrieval with similarity scoring and reranking
- **Document Chunking**: Intelligent text splitting with overlap for better context

### Web Integration
- **Web Scraping**: Extract content from URLs with intelligent parsing
- **Search Integration**: Multiple search providers (DuckDuckGo, Google, Bing, SearX)
- **Content Processing**: Automatic deduplication and cleaning
- **Web-Enhanced Generation**: Augment responses with real-time web data

### File Processing
- **Multi-Format Support**: PDF, DOCX, images, CSV, JSON, HTML, and more
- **OCR Capabilities**: Extract text from images using Tesseract
- **Metadata Extraction**: Comprehensive file information and properties
- **Batch Processing**: Handle multiple files simultaneously

### Advanced Features
- **Parameter Controls**: Fine-tune generation parameters in real-time
- **System Monitoring**: GPU usage, memory, and performance metrics
- **Health Checks**: Comprehensive system status monitoring
- **Caching**: Intelligent caching for improved performance
- **Logging**: Detailed logging with configurable levels

##  Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### Dependencies
All dependencies are listed in `requirements.txt`. Key packages include:
- vLLM Nightly (experimental version)
- Gradio >= 4.0.0
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- ChromaDB >= 0.4.0
- And many more...

##  Installation

### 1. Clone the Repository
```bash
git clone https://https://github.com/cvlt-ai/CVLT-AI-VLLM-WEB-UI-2.0
cd CVLT-AI-VLLM-WEB-UI-2.0
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies (Optional)
For OCR support:
```bash
sudo apt-get install tesseract-ocr  # On Ubuntu/Debian
```

For advanced image processing:
```bash
pip install opencv-python
```

##  Quick Start

### Basic Usage
```bash
python main.py
```

The application will start on `http://localhost:7860`

### Command Line Options
```bash
python main.py --help
```

Available options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 7860)
- `--config`: Path to configuration file
- `--debug`: Enable debug mode
- `--share`: Create public shareable link
- `--model`: Model to load on startup
- `--tensor-parallel-size`: Multi-GPU tensor parallelism
- `--pipeline-parallel-size`: Multi-GPU pipeline parallelism

### Example Commands
```bash
# Start with debug mode
python main.py --debug

# Start on custom port
python main.py --port 8080

# Start with multi-GPU setup
python main.py --tensor-parallel-size 2

# Start with specific model
python main.py --model "microsoft/DialoGPT-medium"
```

##  Project Structure

```
vllm-gradio-webui/
├── src/
│   ├── app_manager.py          # Main application coordinator
│   ├── core/                   # Core vLLM integration
│   │   ├── vllm_manager.py     # vLLM model management
│   │   ├── gpu_manager.py      # GPU resource management
│   │   ├── model_manager.py    # Model discovery and loading
│   │   └── inference_engine.py # Inference coordination
│   ├── rag/                    # RAG system
│   │   ├── pipeline.py         # Main RAG pipeline
│   │   ├── vector_store.py     # Vector database management
│   │   ├── embeddings.py       # Embedding generation
│   │   └── retriever.py        # Document retrieval
│   ├── web/                    # Web functionality
│   │   ├── web_manager.py      # Web integration coordinator
│   │   ├── scraper.py          # Web content scraping
│   │   └── search_integration.py # Search provider integration
│   ├── files/                  # File processing
│   │   ├── file_manager.py     # File upload and management
│   │   └── file_processor.py   # Document processing
│   ├── ui/                     # User interface
│   │   └── gradio_interface.py # Gradio UI components
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration management
│       └── logging.py          # Logging setup
├── config/
│   └── app_config.yaml         # Main configuration file
├── data/                       # Data directories
│   ├── uploads/                # Uploaded files
│   ├── processed/              # Processed content
│   └── metadata/               # File metadata
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── test_basic.py              # Basic functionality tests
└── README.md                  # This file
```

##  Configuration

The application uses a YAML configuration file located at `config/app_config.yaml`. You can customize:

- Model settings and paths
- GPU configuration
- RAG parameters
- Web scraping settings
- File processing options
- UI preferences

Example configuration:
```yaml
vllm:
  model_dirs:
    - "./models"
    - "/path/to/your/models"
  gpu:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9

rag:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

web:
  scraping_timeout: 30
  max_content_length: 100000
  search:
    providers:
      duckduckgo:
        enabled: true
      google:
        enabled: false
        api_key: "your-api-key"
```

##  Usage Guide

### 1. Model Management
- Navigate to the "Models" tab
- Scan directories for available models
- Configure multi-GPU settings
- Load/unload models as needed

### 2. Chat Interface
- Use the main "Chat" tab for conversations
- Select quantization method (GPTQ, AWQ, AutoRound, INT4, INT8, FP8, or none)
- Adjust generation parameters in real-time
- Enable/disable RAG and web search
- Stream responses for better UX

### 3. RAG System
- Upload documents via the "RAG" tab
- Ingest text, files, or web URLs
- Search through your document collection
- Monitor RAG statistics and performance

### 4. File Processing
- Upload files through the "Files" tab
- Support for PDF, Word, images, and more
- Automatic processing and content extraction
- View file details and metadata

### 5. Web Integration
- Search the web via the "Web" tab
- Scrape content from specific URLs
- Enhance responses with web data
- Configure search providers

### 6. System Monitoring
- Monitor GPU usage and memory
- View system performance metrics
- Check application health status
- Access logs and diagnostics

##  Advanced Features

### Multi-GPU Setup
```bash
# 2-GPU tensor parallelism
python main.py --tensor-parallel-size 2

# 4-GPU pipeline parallelism
python main.py --pipeline-parallel-size 4

# Combined setup
python main.py --tensor-parallel-size 2 --pipeline-parallel-size 2
```

### Custom Model Loading
```python
# Load model with custom configuration
config = {
    'tensor_parallel_size': 2,
    'gpu_memory_utilization': 0.8,
    'max_model_len': 4096
}
await app_manager.load_model("your-model-name", config)
```

### RAG Integration
```python
# Ingest documents
await app_manager.ingest_files(["/path/to/document.pdf"])

# Search documents
results = await app_manager.search_documents("your query", top_k=5)

# Generate with RAG
response = await app_manager.generate_response(
    message="your question",
    rag_params={'enabled': True, 'top_k': 5}
)
```

##  Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `gpu_memory_utilization`
   - Use smaller models
   - Enable model quantization (INT8, INT4, or FP8)
   - Try different quantization methods for better compression

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify virtual environment activation

3. **Model Loading Fails**
   - Check model path and permissions
   - Verify model format compatibility
   - Monitor GPU memory usage

4. **Web Scraping Issues**
   - Check internet connectivity
   - Verify target website accessibility
   - Adjust timeout settings

### Debug Mode
```bash
python main.py --debug --log-level DEBUG
```

### Testing
```bash
# Run basic tests
python test_basic.py

# Check specific components
python -c "from src.app_manager import ApplicationManager; print('OK')"
```

##  Performance Tips

1. **GPU Optimization**
   - Use appropriate tensor parallel size
   - Monitor GPU memory utilization
   - Enable mixed precision when possible

2. **RAG Performance**
   - Optimize chunk size for your use case
   - Use efficient embedding models
   - Enable caching for repeated queries

3. **Web Scraping**
   - Implement request rate limiting
   - Use appropriate timeout values
   - Cache frequently accessed content

4. **File Processing**
   - Process files in batches
   - Use appropriate file size limits
   - Enable parallel processing

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for the excellent LLM serving framework
- [Gradio](https://gradio.app/) for the intuitive web interface framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- All other open-source libraries that make this project possible

##  Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

---

For reference testing has been done on a dual gpu system: 1x A6000 Ada & 1x L40 . I am getting about 96-101 toks/s on average having the KV cache stored on gpus, running Qwen 3 30b int8 model with a context size of about 119k. Please let me know of any bugs and I will try to get to them in a timely manner. I HAVE NOT TESTED THIS AT ALL FOR WINDOWS. 

![GitHub Repo stars](https://img.shields.io/github/stars/cvlt-ai/CVLT-AI-VLLM-WEB-UI-2.0?style=social)

