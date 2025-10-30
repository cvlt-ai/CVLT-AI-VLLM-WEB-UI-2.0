#!/usr/bin/env python3
"""
vLLM Gradio WebUI - Main Application Entry Point

Advanced web interface for vLLM with RAG, multi-GPU support, web fetching,
and file processing capabilities.
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app_manager import ApplicationManager
from ui.gradio_interface import GradioInterface

logger = logging.getLogger(__name__)

class VLLMGradioApp:
    """Main application class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.app_manager = None
        self.gradio_interface = None
        self.running = False
    
    async def start(self, **launch_kwargs):
        """Start the application"""
        try:
            logger.info("Starting vLLM Gradio WebUI...")
            
            # Initialize application manager
            self.app_manager = ApplicationManager(self.config_path)
            await self.app_manager.initialize()
            
            # Create Gradio interface
            self.gradio_interface = GradioInterface(self.app_manager)
            interface = self.gradio_interface.create_interface()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.running = True
            logger.info("Application started successfully")
            
            # Launch Gradio interface
            return self.gradio_interface.launch(**launch_kwargs)
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}", exc_info=True)
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the application"""
        try:
            logger.info("Stopping vLLM Gradio WebUI...")
            self.running = False
            
            if self.app_manager:
                await self.app_manager.cleanup()
            
            logger.info("Application stopped")
            
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            # Set a flag to stop the main loop
            self.running = False
            # Create a task to handle cleanup
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop())
            else:
                asyncio.run(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="vLLM Gradio WebUI - Advanced web interface for vLLM"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    # Debug settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to load on startup"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory"
    )
    
    # GPU settings
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU"
    )
    
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size for multi-GPU"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction"
    )
    
    return parser.parse_args()

def setup_logging(level: str = "INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('vllm_gradio_webui.log')
        ]
    )

async def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("vLLM Gradio WebUI Starting")
    logger.info("=" * 60)
    
    try:
        # Create application
        app = VLLMGradioApp(args.config)
        
        # Prepare launch parameters
        launch_params = {
            'server_name': args.host,
            'server_port': args.port,
            'share': args.share,
            'debug': args.debug,
            'show_error': True,
            'quiet': False
        }
        
        # Start application
        demo = await app.start(**launch_params)
        
        logger.info(f"Server running on http://{args.host}:{args.port}")
        
        if args.share:
            logger.info("Public sharing enabled")
        
        # Keep the application running
        try:
            while app.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        
        # Cleanup
        await app.stop()
        
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

