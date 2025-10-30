"""
Configuration management for vLLM Gradio WebUI
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU configuration settings"""
    auto_detect: bool = True
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    enforce_eager: bool = False

@dataclass
class VLLMConfig:
    """vLLM configuration settings"""
    default_model: Optional[str] = None
    model_dirs: list = field(default_factory=lambda: ["./models"])
    gpu: GPUConfig = field(default_factory=GPUConfig)
    
    # Inference parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    stop_sequences: list = field(default_factory=list)
    stream: bool = True
    
    # Performance settings
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: int = 0

@dataclass
class RAGConfig:
    """RAG system configuration"""
    enabled: bool = True
    vector_db_type: str = "chromadb"
    persist_directory: str = "./data/vector_db"
    collection_name: str = "documents"
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"
    
    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = field(default_factory=lambda: ["\n\n", "\n", " ", ""])

@dataclass
class WebScraperConfig:
    """Web scraper configuration"""
    enabled: bool = True
    timeout: int = 30
    max_pages: int = 10
    user_agent: str = "vLLM-Gradio-WebUI/1.0"
    requests_per_minute: int = 60
    delay_between_requests: int = 1

@dataclass
class FileProcessorConfig:
    """File processor configuration"""
    enabled: bool = True
    upload_dir: str = "./data/uploads"
    max_file_size: int = 100  # MB
    allowed_extensions: dict = field(default_factory=lambda: {
        "documents": [".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"],
        "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
        "archives": [".zip", ".tar", ".gz"]
    })
    ocr_enabled: bool = True
    auto_delete: bool = True
    retention_days: int = 7

@dataclass
class UIConfig:
    """UI configuration"""
    theme: str = "default"
    title: str = "vLLM Advanced WebUI"
    description: str = "Comprehensive LLM interface with RAG, multi-GPU support, and advanced features"
    show_gpu_stats: bool = True
    show_model_info: bool = True
    enable_chat_history: bool = True
    max_chat_history: int = 100

@dataclass
class AppConfig:
    """Main application configuration"""
    name: str = "vLLM Gradio WebUI"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    web_scraper: WebScraperConfig = field(default_factory=WebScraperConfig)
    file_processor: FileProcessorConfig = field(default_factory=FileProcessorConfig)
    ui: UIConfig = field(default_factory=UIConfig)

class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        
    def _find_config_file(self) -> str:
        """Find the configuration file"""
        possible_paths = [
            "config/app_config.yaml",
            "../config/app_config.yaml",
            "../../config/app_config.yaml",
            os.path.expanduser("~/.vllm-gradio-webui/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Return default path if none found
        return "config/app_config.yaml"
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return self._parse_config(config_data)
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return AppConfig()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return AppConfig()
    
    def _parse_config(self, config_data: Dict[str, Any]) -> AppConfig:
        """Parse configuration data into AppConfig object"""
        try:
            # Parse app settings
            app_data = config_data.get('app', {})
            
            # Parse vLLM settings
            vllm_data = config_data.get('vllm', {})
            gpu_data = vllm_data.get('gpu', {})
            inference_data = vllm_data.get('inference', {})
            performance_data = vllm_data.get('performance', {})
            
            gpu_config = GPUConfig(
                auto_detect=gpu_data.get('auto_detect', True),
                tensor_parallel_size=gpu_data.get('tensor_parallel_size', 1),
                pipeline_parallel_size=gpu_data.get('pipeline_parallel_size', 1),
                gpu_memory_utilization=gpu_data.get('gpu_memory_utilization', 0.9),
                max_model_len=gpu_data.get('max_model_len'),
                enforce_eager=gpu_data.get('enforce_eager', False)
            )
            
            vllm_config = VLLMConfig(
                default_model=vllm_data.get('default_model'),
                model_dirs=vllm_data.get('model_dirs', ["./models"]),
                gpu=gpu_config,
                temperature=inference_data.get('temperature', 0.7),
                top_p=inference_data.get('top_p', 0.9),
                top_k=inference_data.get('top_k', 50),
                max_tokens=inference_data.get('max_tokens', 512),
                stop_sequences=inference_data.get('stop_sequences', []),
                stream=inference_data.get('stream', True),
                max_num_seqs=performance_data.get('max_num_seqs', 256),
                max_num_batched_tokens=performance_data.get('max_num_batched_tokens'),
                block_size=performance_data.get('block_size', 16),
                swap_space=performance_data.get('swap_space', 4),
                cpu_offload_gb=performance_data.get('cpu_offload_gb', 0)
            )
            
            # Parse RAG settings
            rag_data = config_data.get('rag', {})
            vector_db_data = rag_data.get('vector_db', {})
            embeddings_data = rag_data.get('embeddings', {})
            retrieval_data = rag_data.get('retrieval', {})
            chunking_data = rag_data.get('chunking', {})
            
            rag_config = RAGConfig(
                enabled=rag_data.get('enabled', True),
                vector_db_type=vector_db_data.get('type', 'chromadb'),
                persist_directory=vector_db_data.get('persist_directory', './data/vector_db'),
                collection_name=vector_db_data.get('collection_name', 'documents'),
                embedding_model=embeddings_data.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                embedding_device=embeddings_data.get('device', 'auto'),
                top_k=retrieval_data.get('top_k', 5),
                similarity_threshold=retrieval_data.get('similarity_threshold', 0.7),
                max_context_length=retrieval_data.get('max_context_length', 4000),
                chunk_size=chunking_data.get('chunk_size', 1000),
                chunk_overlap=chunking_data.get('chunk_overlap', 200),
                separators=chunking_data.get('separators', ["\n\n", "\n", " ", ""])
            )
            
            # Parse other component configs
            web_scraper_data = config_data.get('web_scraper', {})
            web_scraper_config = WebScraperConfig(
                enabled=web_scraper_data.get('enabled', True),
                timeout=web_scraper_data.get('timeout', 30),
                max_pages=web_scraper_data.get('max_pages', 10),
                user_agent=web_scraper_data.get('user_agent', 'vLLM-Gradio-WebUI/1.0'),
                requests_per_minute=web_scraper_data.get('rate_limit', {}).get('requests_per_minute', 60),
                delay_between_requests=web_scraper_data.get('rate_limit', {}).get('delay_between_requests', 1)
            )
            
            file_processor_data = config_data.get('file_processor', {})
            file_processor_config = FileProcessorConfig(
                enabled=file_processor_data.get('enabled', True),
                upload_dir=file_processor_data.get('upload_dir', './data/uploads'),
                max_file_size=file_processor_data.get('max_file_size', 100),
                allowed_extensions=file_processor_data.get('allowed_extensions', {}),
                ocr_enabled=file_processor_data.get('ocr', {}).get('enabled', True),
                auto_delete=file_processor_data.get('cleanup', {}).get('auto_delete', True),
                retention_days=file_processor_data.get('cleanup', {}).get('retention_days', 7)
            )
            
            ui_data = config_data.get('ui', {})
            ui_config = UIConfig(
                theme=ui_data.get('theme', 'default'),
                title=ui_data.get('title', 'vLLM Advanced WebUI'),
                description=ui_data.get('description', 'Comprehensive LLM interface'),
                show_gpu_stats=ui_data.get('interface', {}).get('show_gpu_stats', True),
                show_model_info=ui_data.get('interface', {}).get('show_model_info', True),
                enable_chat_history=ui_data.get('interface', {}).get('enable_chat_history', True),
                max_chat_history=ui_data.get('interface', {}).get('max_chat_history', 100)
            )
            
            return AppConfig(
                name=app_data.get('name', 'vLLM Gradio WebUI'),
                version=app_data.get('version', '1.0.0'),
                host=app_data.get('host', '0.0.0.0'),
                port=app_data.get('port', 7860),
                debug=app_data.get('debug', False),
                log_level=app_data.get('log_level', 'INFO'),
                vllm=vllm_config,
                rag=rag_config,
                web_scraper=web_scraper_config,
                file_processor=file_processor_config,
                ui=ui_config
            )
            
        except Exception as e:
            logger.error(f"Error parsing config: {e}")
            return AppConfig()
    
    def save_config(self, config: AppConfig):
        """Save configuration to file"""
        try:
            # Convert config to dict for YAML serialization
            config_dict = self._config_to_dict(config)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to dictionary for YAML serialization"""
        # This is a simplified version - in practice, you'd want to implement
        # proper serialization for all nested dataclasses
        return {
            'app': {
                'name': config.name,
                'version': config.version,
                'host': config.host,
                'port': config.port,
                'debug': config.debug,
                'log_level': config.log_level
            }
            # Add other sections as needed
        }
    
    def get_config(self) -> AppConfig:
        """Get the current configuration"""
        return self.config
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        logger.info("Configuration reloaded")

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the global configuration"""
    return config_manager.get_config()

def reload_config():
    """Reload the global configuration"""
    config_manager.reload_config()

