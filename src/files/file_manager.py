"""
File Manager for vLLM Gradio WebUI

Handles file uploads, storage, metadata management, and integration
with RAG system for document processing.
"""

import logging
import os
import shutil
import time
import hashlib
import json
from typing import List, Dict, Optional, Any, BinaryIO
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile

from .file_processor import FileProcessor, FileInfo, ProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class UploadedFile:
    """Uploaded file information"""
    file_id: str
    original_filename: str
    stored_filename: str
    file_path: str
    file_size: int
    mime_type: str
    upload_timestamp: float
    processed: bool = False
    processing_error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class FileManagerConfig:
    """Configuration for file manager"""
    upload_dir: str = "./data/uploads"
    processed_dir: str = "./data/processed"
    metadata_dir: str = "./data/metadata"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = None
    auto_process: bool = True
    keep_originals: bool = True

class FileManager:
    """Manages file uploads and processing"""
    
    def __init__(self, config: FileManagerConfig = None, 
                 file_processor: FileProcessor = None,
                 rag_pipeline=None):
        self.config = config or FileManagerConfig()
        self.file_processor = file_processor or FileProcessor()
        self.rag_pipeline = rag_pipeline
        
        # Create directories
        self._create_directories()
        
        # Load existing file metadata
        self.uploaded_files: Dict[str, UploadedFile] = {}
        self._load_metadata()
        
        logger.info("File manager initialized")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.upload_dir,
            self.config.processed_dir,
            self.config.metadata_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_metadata(self):
        """Load existing file metadata"""
        try:
            metadata_file = os.path.join(self.config.metadata_dir, "files.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                for file_id, file_data in data.items():
                    self.uploaded_files[file_id] = UploadedFile(**file_data)
                
                logger.info(f"Loaded metadata for {len(self.uploaded_files)} files")
                
        except Exception as e:
            logger.warning(f"Failed to load file metadata: {e}")
    
    def _save_metadata(self):
        """Save file metadata"""
        try:
            metadata_file = os.path.join(self.config.metadata_dir, "files.json")
            
            data = {}
            for file_id, uploaded_file in self.uploaded_files.items():
                data[file_id] = asdict(uploaded_file)
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")
    
    def _generate_file_id(self, filename: str, content_hash: str = None) -> str:
        """Generate unique file ID"""
        timestamp = str(int(time.time()))
        
        if content_hash:
            hash_input = f"{filename}:{content_hash}:{timestamp}"
        else:
            hash_input = f"{filename}:{timestamp}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _get_content_hash(self, file_path: str) -> str:
        """Calculate content hash for file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate content hash: {e}")
            return ""
    
    async def upload_file(self, file_data: BinaryIO, filename: str, 
                         mime_type: str = None) -> Dict[str, Any]:
        """Upload and store a file"""
        try:
            # Validate filename
            if not filename or filename.strip() == "":
                return {
                    'success': False,
                    'error': 'Invalid filename'
                }
            
            # Check file extension
            file_ext = Path(filename).suffix.lower()
            if not self.file_processor.is_supported_file(filename):
                return {
                    'success': False,
                    'error': f'File type not supported: {file_ext}'
                }
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_path = temp_file.name
            
            try:
                # Write file data
                file_size = 0
                while True:
                    chunk = file_data.read(8192)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    file_size += len(chunk)
                    
                    # Check size limit
                    if file_size > self.config.max_file_size:
                        temp_file.close()
                        os.unlink(temp_path)
                        return {
                            'success': False,
                            'error': f'File too large: {file_size} bytes'
                        }
                
                temp_file.close()
                
                # Calculate content hash
                content_hash = self._get_content_hash(temp_path)
                
                # Check for duplicates
                duplicate_file = self._find_duplicate(content_hash)
                if duplicate_file:
                    os.unlink(temp_path)
                    return {
                        'success': True,
                        'file_id': duplicate_file.file_id,
                        'message': 'File already exists',
                        'duplicate': True
                    }
                
                # Generate file ID and stored filename
                file_id = self._generate_file_id(filename, content_hash)
                stored_filename = f"{file_id}_{filename}"
                final_path = os.path.join(self.config.upload_dir, stored_filename)
                
                # Move file to final location
                shutil.move(temp_path, final_path)
                
                # Create uploaded file record
                uploaded_file = UploadedFile(
                    file_id=file_id,
                    original_filename=filename,
                    stored_filename=stored_filename,
                    file_path=final_path,
                    file_size=file_size,
                    mime_type=mime_type or "",
                    upload_timestamp=time.time(),
                    metadata={'content_hash': content_hash}
                )
                
                # Store in memory and save metadata
                self.uploaded_files[file_id] = uploaded_file
                self._save_metadata()
                
                # Auto-process if enabled
                if self.config.auto_process:
                    await self.process_file(file_id)
                
                logger.info(f"File uploaded successfully: {filename} -> {file_id}")
                
                return {
                    'success': True,
                    'file_id': file_id,
                    'filename': filename,
                    'size': file_size,
                    'processed': uploaded_file.processed
                }
                
            except Exception as e:
                # Cleanup on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
                
        except Exception as e:
            logger.error(f"File upload failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def upload_file_from_path(self, file_path: str) -> Dict[str, Any]:
        """Upload file from local path"""
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': 'File not found'
                }
            
            filename = os.path.basename(file_path)
            
            with open(file_path, 'rb') as f:
                return await self.upload_file(f, filename)
                
        except Exception as e:
            logger.error(f"Failed to upload file from path: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _find_duplicate(self, content_hash: str) -> Optional[UploadedFile]:
        """Find duplicate file by content hash"""
        for uploaded_file in self.uploaded_files.values():
            if uploaded_file.metadata and uploaded_file.metadata.get('content_hash') == content_hash:
                return uploaded_file
        return None
    
    async def process_file(self, file_id: str) -> Dict[str, Any]:
        """Process an uploaded file"""
        try:
            uploaded_file = self.uploaded_files.get(file_id)
            if not uploaded_file:
                return {
                    'success': False,
                    'error': 'File not found'
                }
            
            if uploaded_file.processed:
                return {
                    'success': True,
                    'message': 'File already processed',
                    'file_id': file_id
                }
            
            # Process the file
            file_info = await self.file_processor.process_file(uploaded_file.file_path)
            
            if file_info.success:
                # Update uploaded file record
                uploaded_file.processed = True
                uploaded_file.processing_error = None
                uploaded_file.metadata.update({
                    'processing_time': file_info.processing_time,
                    'content_length': len(file_info.content),
                    'file_type': file_info.file_type,
                    'extracted_metadata': file_info.metadata
                })
                
                # Save processed content
                processed_file_path = os.path.join(
                    self.config.processed_dir, 
                    f"{file_id}_content.txt"
                )
                
                with open(processed_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_info.content)
                
                # Save metadata
                metadata_file_path = os.path.join(
                    self.config.metadata_dir,
                    f"{file_id}_metadata.json"
                )
                
                with open(metadata_file_path, 'w') as f:
                    json.dump(file_info.metadata, f, indent=2)
                
                # Ingest into RAG if available
                if self.rag_pipeline:
                    await self._ingest_to_rag(file_id, file_info)
                
                # Save updated metadata
                self._save_metadata()
                
                logger.info(f"File processed successfully: {file_id}")
                
                return {
                    'success': True,
                    'file_id': file_id,
                    'content_length': len(file_info.content),
                    'processing_time': file_info.processing_time
                }
            else:
                # Update with error
                uploaded_file.processing_error = file_info.error_message
                self._save_metadata()
                
                return {
                    'success': False,
                    'error': file_info.error_message,
                    'file_id': file_id
                }
                
        except Exception as e:
            logger.error(f"File processing failed: {e}", exc_info=True)
            
            # Update with error
            if file_id in self.uploaded_files:
                self.uploaded_files[file_id].processing_error = str(e)
                self._save_metadata()
            
            return {
                'success': False,
                'error': str(e),
                'file_id': file_id
            }
    
    async def _ingest_to_rag(self, file_id: str, file_info: FileInfo):
        """Ingest processed file into RAG system"""
        try:
            uploaded_file = self.uploaded_files[file_id]
            
            # Prepare document for RAG
            document = {
                'id': f"file_{file_id}",
                'content': file_info.content,
                'type': 'file',
                'path': uploaded_file.file_path,
                'metadata': {
                    'file_id': file_id,
                    'original_filename': uploaded_file.original_filename,
                    'file_type': file_info.file_type,
                    'file_size': uploaded_file.file_size,
                    'upload_timestamp': uploaded_file.upload_timestamp,
                    'processing_time': file_info.processing_time,
                    **file_info.metadata
                }
            }
            
            # Ingest into RAG
            result = await self.rag_pipeline.ingest_documents([document])
            
            if result.success:
                # Update metadata with RAG info
                uploaded_file.metadata['rag_ingested'] = True
                uploaded_file.metadata['rag_chunks'] = result.chunks_created
                logger.info(f"File {file_id} ingested into RAG system")
            else:
                uploaded_file.metadata['rag_ingested'] = False
                uploaded_file.metadata['rag_error'] = result.error_message
                logger.warning(f"Failed to ingest file {file_id} into RAG: {result.error_message}")
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"RAG ingestion failed for file {file_id}: {e}")
    
    async def process_all_files(self) -> Dict[str, Any]:
        """Process all unprocessed files"""
        unprocessed_files = [
            file_id for file_id, uploaded_file in self.uploaded_files.items()
            if not uploaded_file.processed
        ]
        
        if not unprocessed_files:
            return {
                'success': True,
                'message': 'No files to process',
                'processed_count': 0
            }
        
        results = []
        for file_id in unprocessed_files:
            result = await self.process_file(file_id)
            results.append(result)
        
        successful_count = sum(1 for r in results if r['success'])
        
        return {
            'success': True,
            'processed_count': successful_count,
            'total_count': len(unprocessed_files),
            'results': results
        }
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an uploaded file"""
        uploaded_file = self.uploaded_files.get(file_id)
        if not uploaded_file:
            return None
        
        info = asdict(uploaded_file)
        
        # Add processed content if available
        if uploaded_file.processed:
            processed_file_path = os.path.join(
                self.config.processed_dir,
                f"{file_id}_content.txt"
            )
            
            if os.path.exists(processed_file_path):
                try:
                    with open(processed_file_path, 'r', encoding='utf-8') as f:
                        info['content'] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read processed content: {e}")
        
        return info
    
    def get_file_content(self, file_id: str) -> Optional[str]:
        """Get processed content of a file"""
        uploaded_file = self.uploaded_files.get(file_id)
        if not uploaded_file or not uploaded_file.processed:
            return None
        
        processed_file_path = os.path.join(
            self.config.processed_dir,
            f"{file_id}_content.txt"
        )
        
        try:
            with open(processed_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file content: {e}")
            return None
    
    def list_files(self, processed_only: bool = False) -> List[Dict[str, Any]]:
        """List all uploaded files"""
        files = []
        
        for file_id, uploaded_file in self.uploaded_files.items():
            if processed_only and not uploaded_file.processed:
                continue
            
            file_info = {
                'file_id': file_id,
                'filename': uploaded_file.original_filename,
                'size': uploaded_file.file_size,
                'upload_time': uploaded_file.upload_timestamp,
                'processed': uploaded_file.processed,
                'processing_error': uploaded_file.processing_error,
                'mime_type': uploaded_file.mime_type
            }
            
            if uploaded_file.metadata:
                file_info.update({
                    'content_length': uploaded_file.metadata.get('content_length', 0),
                    'file_type': uploaded_file.metadata.get('file_type', ''),
                    'rag_ingested': uploaded_file.metadata.get('rag_ingested', False)
                })
            
            files.append(file_info)
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return files
    
    def delete_file(self, file_id: str) -> Dict[str, Any]:
        """Delete an uploaded file"""
        try:
            uploaded_file = self.uploaded_files.get(file_id)
            if not uploaded_file:
                return {
                    'success': False,
                    'error': 'File not found'
                }
            
            # Delete original file
            if os.path.exists(uploaded_file.file_path):
                os.unlink(uploaded_file.file_path)
            
            # Delete processed content
            processed_file_path = os.path.join(
                self.config.processed_dir,
                f"{file_id}_content.txt"
            )
            if os.path.exists(processed_file_path):
                os.unlink(processed_file_path)
            
            # Delete metadata file
            metadata_file_path = os.path.join(
                self.config.metadata_dir,
                f"{file_id}_metadata.json"
            )
            if os.path.exists(metadata_file_path):
                os.unlink(metadata_file_path)
            
            # Remove from memory
            del self.uploaded_files[file_id]
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"File deleted: {file_id}")
            
            return {
                'success': True,
                'file_id': file_id
            }
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file manager statistics"""
        total_files = len(self.uploaded_files)
        processed_files = sum(1 for f in self.uploaded_files.values() if f.processed)
        total_size = sum(f.file_size for f in self.uploaded_files.values())
        
        # File type distribution
        file_types = {}
        for uploaded_file in self.uploaded_files.values():
            file_type = uploaded_file.metadata.get('file_type', 'unknown') if uploaded_file.metadata else 'unknown'
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_files': total_files,
            'processed_files': processed_files,
            'unprocessed_files': total_files - processed_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_types': file_types,
            'upload_dir': self.config.upload_dir,
            'processed_dir': self.config.processed_dir
        }
    
    def cleanup_old_files(self, days: int = 30) -> Dict[str, Any]:
        """Clean up files older than specified days"""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            deleted_files = []
            
            for file_id, uploaded_file in list(self.uploaded_files.items()):
                if uploaded_file.upload_timestamp < cutoff_time:
                    result = self.delete_file(file_id)
                    if result['success']:
                        deleted_files.append(file_id)
            
            return {
                'success': True,
                'deleted_count': len(deleted_files),
                'deleted_files': deleted_files
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

