#  CVLT AI vLLM Gradio WebUI Architecture

## Overview
This is a comprehensive web-based interface for vLLM that provides advanced features including multi-GPU support, RAG capabilities, web scraping, and file processing.

## System Architecture

### Core Components

#### 1. vLLM Integration Layer (`src/core/`)
- **vllm_manager.py**: Main vLLM server management and configuration
- **model_manager.py**: Model discovery, loading, and switching
- **gpu_manager.py**: Multi-GPU configuration and resource management
- **inference_engine.py**: Inference request handling and response processing

#### 2. RAG System (`src/rag/`)
- **vector_store.py**: Vector database management (ChromaDB/FAISS)
- **embeddings.py**: Document embedding and retrieval
- **retriever.py**: Context retrieval and ranking
- **rag_pipeline.py**: End-to-end RAG processing

#### 3. Web Scraping Module (`src/web_scraper/`)
- **scraper.py**: Web content extraction using BeautifulSoup
- **url_processor.py**: URL validation and content processing
- **search_integration.py**: Web search capabilities

#### 4. File Processing Module (`src/file_processor/`)
- **document_processor.py**: PDF, DOCX, TXT processing
- **image_processor.py**: Image analysis and OCR
- **file_manager.py**: Upload handling and file management

#### 5. User Interface (`src/ui/`)
- **gradio_app.py**: Main Gradio application
- **components.py**: Custom UI components
- **layouts.py**: UI layout definitions
- **callbacks.py**: Event handlers and callbacks

#### 6. Utilities (`src/utils/`)
- **config.py**: Configuration management
- **logging.py**: Logging setup
- **helpers.py**: Common utility functions

### Multi-GPU Support

#### Tensor Parallelism
- Distributes model layers across multiple GPUs on a single node
- Configurable via `tensor_parallel_size` parameter
- Supports 2, 4, 8, or 16 GPUs

#### Pipeline Parallelism
- Distributes model layers across multiple nodes
- Configurable via `pipeline_parallel_size` parameter
- Combined with tensor parallelism for large deployments

#### GPU Resource Management
- Automatic GPU detection and configuration
- Memory monitoring and optimization
- Dynamic resource allocation

### Data Flow

1. **User Input** → Gradio Interface
2. **Request Processing** → Core Engine
3. **Context Retrieval** (if RAG enabled) → Vector Store
4. **Model Inference** → vLLM Engine (Multi-GPU)
5. **Response Generation** → User Interface

### Configuration Management

#### Model Configuration (`config/models/`)
- Model discovery and metadata
- GPU requirements and optimization settings
- Model-specific parameters

#### GPU Configuration (`config/gpu/`)
- Multi-GPU topology
- Parallelism settings
- Resource allocation policies

### Advanced Features

#### 1. Model Selection by Folder
- Automatic model discovery from specified directories
- Support for HuggingFace models and local models
- Model metadata extraction and caching

#### 2. RAG Integration
- Document ingestion and embedding
- Vector similarity search
- Context-aware response generation

#### 3. Web Data Fetching
- Real-time web scraping
- Search engine integration
- Content extraction and processing

#### 4. File Upload and Processing
- Multi-format document support
- Image analysis and OCR
- Automatic content extraction

#### 5. Parameter Controls
- Temperature, top-p, top-k controls
- Max tokens and stop sequences
- Streaming response options

#### 6. Advanced UI Features
- Chat history management
- Export/import conversations
- Theme customization
- Real-time GPU monitoring

## Deployment Architecture

### Single Node Deployment
```
[Gradio UI] → [vLLM Engine] → [Multi-GPU (Tensor Parallel)]
     ↓              ↓
[RAG System] → [Vector DB]
     ↓
[File Processor] → [Web Scraper]
```

### Multi-Node Deployment
```
Node 1: [Gradio UI] → [vLLM Coordinator]
Node 2-N: [vLLM Workers] → [GPU Arrays (Pipeline Parallel)]
Shared: [Vector DB] → [File Storage]
```

## Technology Stack

- **Backend**: Python 3.11+, vLLM, FastAPI
- **Frontend**: Gradio
- **Vector DB**: ChromaDB (primary), FAISS (alternative)
- **ML Libraries**: Transformers, Sentence-Transformers
- **Web Scraping**: BeautifulSoup4, Requests
- **File Processing**: PyPDF2, python-docx, Pillow
- **GPU**: CUDA, PyTorch
- **Deployment**: Docker, Ray (for multi-node)

## Security Considerations

- Input validation and sanitization
- File upload restrictions and scanning
- Rate limiting for API endpoints
- Secure model file handling
- Network security for multi-node deployments

