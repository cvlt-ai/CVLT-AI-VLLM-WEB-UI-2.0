#!/usr/bin/env python3
"""
Basic test script for vLLM Gradio WebUI

Tests basic functionality without starting the full application.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all main modules can be imported"""
    print("Testing imports...")
    
    try:
        from app_manager import ApplicationManager
        print("✓ ApplicationManager imported")
    except Exception as e:
        print(f"✗ ApplicationManager failed: {e}")
        return False
    
    try:
        from ui.gradio_interface import GradioInterface
        print("✓ GradioInterface imported")
    except Exception as e:
        print(f"✗ GradioInterface failed: {e}")
        return False
    
    try:
        from core.vllm_manager import VLLMManager
        print("✓ VLLMManager imported")
    except Exception as e:
        print(f"✗ VLLMManager failed: {e}")
        return False
    
    try:
        from rag.pipeline import RAGPipeline
        print("✓ RAGPipeline imported")
    except Exception as e:
        print(f"✗ RAGPipeline failed: {e}")
        return False
    
    try:
        from web.web_manager import WebManager
        print("✓ WebManager imported")
    except Exception as e:
        print(f"✗ WebManager failed: {e}")
        return False
    
    try:
        from files.file_manager import FileManager
        print("✓ FileManager imported")
    except Exception as e:
        print(f"✗ FileManager failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print("✓ Configuration loaded")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def test_directories():
    """Test that required directories can be created"""
    print("\nTesting directory creation...")
    
    test_dirs = [
        "./data/uploads",
        "./data/processed", 
        "./data/metadata",
        "./logs"
    ]
    
    try:
        for directory in test_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        return True
    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without vLLM"""
    print("\nTesting basic functionality...")
    
    try:
        # Test file processor
        from files.file_processor import FileProcessor, ProcessingConfig
        config = ProcessingConfig()
        processor = FileProcessor(config)
        print("✓ FileProcessor created")
        
        # Test web scraper
        from web.scraper import WebScraper, ScrapingConfig
        scraper_config = ScrapingConfig()
        scraper = WebScraper(scraper_config)
        print("✓ WebScraper created")
        
        # Test search integration
        from web.search_integration import SearchManager
        search_config = {'providers': {'duckduckgo': {'enabled': True}}}
        search_manager = SearchManager(search_config)
        print("✓ SearchManager created")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("vLLM Gradio WebUI - Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_directories,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! The application should work correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

