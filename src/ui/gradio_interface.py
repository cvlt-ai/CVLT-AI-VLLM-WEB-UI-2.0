"""
Gradio Interface for vLLM Gradio WebUI

Main Gradio interface with advanced UI components including model selection,
RAG controls, file upload, web fetching, and parameter adjustment.
"""

import logging
import gradio as gr
import asyncio
import json
import time
from typing import List, Dict, Optional, Any, Tuple
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class GradioInterface:
    """Main Gradio interface for the vLLM WebUI"""
    
    def __init__(self, app_manager):
        self.app_manager = app_manager
        self.interface = None
        
        # UI state
        self.current_model = None
        self.conversation_history = []
        self.context_tokens_used = 0
        self.max_context_length = 32768  # Default, will be updated when model loads
        
        logger.info("Gradio interface initialized")
        
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        
        # Custom CSS for monochromatic clean appearance
        custom_css = """
/* Global font family - clean sans-serif */
* {
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
}

/* Main container styling */
.main-container {
max-width: 1400px;
margin: 0 auto;
}

/* Chat container */
.chat-container {
height: 600px;
overflow-y: auto;
}

/* Monochromatic color scheme - grays and blacks */
.parameter-panel {
background: #f5f5f5;
padding: 15px;
border-radius: 4px;
margin: 10px 0;
border: 1px solid #e0e0e0;
}

/* Status panel */
.status-panel {
background: #fafafa;
padding: 10px;
border-radius: 4px;
margin: 5px 0;
border: 1px solid #e0e0e0;
}

/* Website status */
.website-status {
background: #f5f5f5;
padding: 8px;
border-radius: 4px;
margin: 5px 0;
border: 1px solid #d0d0d0;
}

/* Model info */
.model-info {
background: #fafafa;
padding: 10px;
border-radius: 4px;
margin: 5px 0;
border: 1px solid #e0e0e0;
}

/* Clean button styling */
.gradio-button {
border-radius: 4px !important;
}

/* Clean input styling */
.gradio-textbox, .gradio-dropdown {
border-radius: 4px !important;
}
"""
        
        with gr.Blocks(
            title="vLLM Gradio WebUI",
            theme=gr.themes.Monochrome(),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown(
                """
# CVLT AI vLLM Gradio WebUI

Advanced web interface for vLLM with RAG, web fetching, and file processing capabilities.
""",
                elem_classes=["main-container"]
            )
            
            # Main layout with tabs
            with gr.Tabs():
                
                # Chat Tab
                with gr.Tab("Chat", id="chat"):
                    self._create_chat_tab()
                
                # Model Management Tab
                with gr.Tab("Models", id="models"):
                    self._create_model_tab()
                
                # RAG Tab
                with gr.Tab("RAG", id="rag"):
                    self._create_rag_tab()
                
                # Files Tab
                with gr.Tab("Files", id="files"):
                    self._create_files_tab()
                
                # Web Tab
                with gr.Tab("Web", id="web"):
                    self._create_web_tab()
                
                # Settings Tab
                with gr.Tab("Settings", id="settings"):
                    self._create_settings_tab()
                
                # System Tab
                with gr.Tab("System", id="system"):
                    self._create_system_tab()
                
                self.interface = interface
                return interface
    
    def _create_chat_tab(self):
        """Create the main chat interface"""
        
        with gr.Row():
            # Left column - Chat
            with gr.Column(scale=3):
                
                # Model selection and status
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=self._get_available_models(),
                        value=None,
                        interactive=True,
                        elem_id="model_selector"
                    )
                    
                    refresh_models_btn = gr.Button("Refresh",
                        size="sm",
                        variant="secondary"
                    )
                
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    elem_classes=["chat-container"],
                    show_copy_button=True,
                    bubble_full_width=False,
                    type="messages"
                )
                
                # Input area
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=3,
                        scale=4
                    )
                    
                    with gr.Column(scale=1, min_width=100):
                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                            size="sm"
                        )
                
                # Chat options
                with gr.Row():
                    use_rag = gr.Checkbox(
                        label="Use RAG",
                        value=False,
                        info="Enable retrieval-augmented generation"
                    )
                    
                    use_web = gr.Checkbox(
                        label="Web Search",
                        value=False,
                        info="Search web for additional context"
                    )
                    
                    thinking_mode = gr.Checkbox(
                        label="Thinking Mode",
                        value=False,
                        info="Enable thinking mode for reasoning models (o1, QwQ, etc.)"
                    )
                
                # Website input section
                with gr.Accordion("Website Context", open=False):
                    with gr.Row():
                        website_urls = gr.Textbox(
                            label="Website URLs",
                            placeholder="Enter URLs separated by commas (e.g., https://example.com, https://docs.example.com)",
                            lines=2,
                            info="Add website content to conversation context"
                        )
                        
                        visit_websites_btn = gr.Button("Visit Websites",
                            variant="secondary",
                            size="sm"
                        )
                        
                        website_status = gr.Markdown(
                            "No websites visited",
                            elem_classes=["website-status"]
                        )
                
                # Right column - Parameters and controls
                with gr.Column(scale=1):
                    
                    # Model Configuration
                    with gr.Accordion("Model Configuration", open=False):
                        context_length = gr.Slider(
                            minimum=1024,
                            maximum=262144,
                            value=32768,
                            step=1024,
                            label="Context Length",
                            info="Maximum sequence length (affects memory usage)"
                        )
                        
                        gpu_memory_util = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.9,
                            step=0.05,
                            label="GPU Memory Utilization",
                            info="Fraction of GPU memory to use"
                        )
                        
                        kv_cache_dtype = gr.Dropdown(
                            choices=["auto", "fp16", "fp8", "int8"],
                            value="auto",
                            label="KV Cache Data Type",
                            info="Data type for KV cache (affects memory usage)"
                        )
                        
                        cpu_offload_gb = gr.Slider(
                            minimum=0,
                            maximum=64,
                            value=0,
                            step=1,
                            label="CPU Offload (GB)",
                            info="Amount of KV cache to offload to CPU memory"
                        )
                        
                        quantization = gr.Dropdown(
                            choices=["none", "gptq", "awq", "autoround", "int4", "int8", "fp8"],
                            value="none",
                            label="Quantization",
                            info="Quantization method for model compression"
                        )
                        
                        enforce_eager = gr.Checkbox(
                            value=False,
                            label="Enforce Eager Mode",
                            info="Disable CUDA graphs for debugging"
                        )
                    
                    # Generation parameters
                    with gr.Accordion("Generation Parameters", open=False):
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness"
                        )
                        
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p",
                            info="Nucleus sampling"
                        )
                        
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top-k",
                            info="Top-k sampling"
                        )
                        
                        max_tokens = gr.Slider(
                            minimum=1,
                            maximum=24576,
                            value=512,
                            step=1,
                            label="Max Tokens",
                            info="Maximum response length"
                        )
                        
                        repetition_penalty = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.1,
                            step=0.1,
                            label="Repetition Penalty",
                            info="Penalize repetition"
                        )
                    
                    # RAG parameters
                    with gr.Accordion("RAG Parameters", open=False):
                        rag_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="RAG Top-k",
                            info="Number of documents to retrieve"
                        )
                        
                        rag_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="RAG Threshold",
                            info="Minimum similarity score"
                        )
                    
                    # Web search parameters
                    with gr.Accordion("Web Parameters", open=False):
                        web_max_results = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Web Results",
                            info="Max web search results"
                        )
                        
                        web_provider = gr.Dropdown(
                            label="Search Provider",
                            choices=["auto", "duckduckgo", "google", "bing"],
                            value="auto",
                            info="Web search provider"
                        )
                    
                    # Event handlers
                    def load_model(model_name, context_len, gpu_mem_util, kv_dtype, cpu_offload, quant_method, eager_mode):
                        if not model_name:
                            return "No model selected"
                        
                        try:
                            # Prepare model configuration
                            config = {
                                'max_model_len': int(context_len),
                                'context_length': int(context_len),
                                'gpu_memory_utilization': float(gpu_mem_util),
                                'kv_cache_dtype': kv_dtype,
                                'cpu_offload_gb': int(cpu_offload),
                                'quantization': quant_method if quant_method != 'none' else None,
                                'enforce_eager': bool(eager_mode)
                            }
                            
                            result = asyncio.run(self.app_manager.load_model(model_name, config))
                            if result['success']:
                                self.current_model = model_name
                                return f" Model loaded: {model_name}\n Context: {context_len:,} tokens\n GPU Memory: {gpu_mem_util:.1%}\n KV Cache: {kv_dtype}\n CPU Offload: {cpu_offload}GB"
                            else:
                                return f" Failed to load model: {result.get('error', 'Unknown error')}"
                        except Exception as e:
                            return f" Error loading model: {str(e)}"
                    
                    def refresh_models():
                        models = self._get_available_models()
                        return gr.Dropdown(choices=models)
                    
                    def send_message(message, history, model, use_rag_flag, use_web_flag, 
                                    temp, top_p_val, top_k_val, max_tok, rep_penalty,
                                    rag_k, rag_thresh, web_results, web_prov, thinking_mode_flag):
                        
                        if not message.strip():
                            return history, ""
                        
                        if not model:
                            # Add user message
                            history.append({"role": "user", "content": message})
                            # Add error response
                            history.append({"role": "assistant", "content": " Please select a model first."})
                            return history, ""
                        
                        try:
                            # Prepare generation parameters
                            gen_params = {
                                'temperature': temp,
                                'top_p': top_p_val,
                                'top_k': top_k_val,
                                'max_tokens': max_tok,
                                'repetition_penalty': rep_penalty,
                                'thinking_mode': thinking_mode_flag
                            }
                            
                            # Prepare RAG parameters
                            rag_params = {
                                'enabled': use_rag_flag,
                                'top_k': rag_k,
                                'threshold': rag_thresh
                            }
                            
                            # Prepare web parameters
                            web_params = {
                                'enabled': use_web_flag,
                                'max_results': web_results,
                                'provider': web_prov if web_prov != "auto" else None
                            }
                            
                            # Determine if we should use simple mode
                            # Use enhanced mode if RAG or web features are enabled
                            use_simple_mode = not (use_rag_flag or use_web_flag)
                            
                            # Add website context if available
                            enhanced_message = message
                            if hasattr(self, 'website_context') and self.website_context:
                                website_content = []
                                for site in self.website_context[-3:]:  # Use last 3 websites
                                    website_content.append(f"Website: {site['title']} ({site['url']})\nContent: {site['content'][:2000]}...")
                                
                                if website_content:
                                    website_context_text = "\n\n".join(website_content)
                                    enhanced_message = f"Website Context:\n{website_context_text}\n\nUser Question: {message}"
                                # Don't use simple mode if we have website context
                                use_simple_mode = False
                            
                            # Generate response - streaming is currently disabled for stability
                            response = asyncio.run(self.app_manager.generate_response(
                                message=enhanced_message,
                                generation_params=gen_params,
                                rag_params=rag_params,
                                web_params=web_params,
                                stream=False,  # Streaming disabled for stability with vLLM async engine
                                simple_mode=use_simple_mode
                            ))
                            
                            if response['success']:
                                # Add user message
                                history.append({"role": "user", "content": message})
                                # Add assistant response
                                history.append({"role": "assistant", "content": response.get('text', response.get('response', 'No response'))})
                            else:
                                # Add user message
                                history.append({"role": "user", "content": message})
                                # Add error response
                                history.append({"role": "assistant", "content": f" Error: {response.get('error', 'Unknown error')}"})
                            
                            return history, ""
                        
                        except Exception as e:
                            # Add user message
                            history.append({"role": "user", "content": message})
                            # Add error response
                            history.append({"role": "assistant", "content": f" Error: {str(e)}"})
                            return history, ""
                    
                    def clear_chat():
                        self.conversation_history = []
                        return []
                    
                    # Connect events
                    # Model dropdown change event removed since model_status component was removed
                    
                    refresh_models_btn.click(
                        fn=refresh_models,
                        outputs=[model_dropdown]
                    )
                    
                    send_btn.click(
                        fn=send_message,
                        inputs=[
                            msg_input, chatbot, model_dropdown, use_rag, use_web,
                            temperature, top_p, top_k, max_tokens, repetition_penalty,
                            rag_top_k, rag_threshold, web_max_results, web_provider, thinking_mode
                        ],
                        outputs=[chatbot, msg_input]
                    )
                    
                    msg_input.submit(
                        fn=send_message,
                        inputs=[
                            msg_input, chatbot, model_dropdown, use_rag, use_web,
                            temperature, top_p, top_k, max_tokens, repetition_penalty,
                            rag_top_k, rag_threshold, web_max_results, web_provider, thinking_mode
                        ],
                        outputs=[chatbot, msg_input]
                    )
                    
                    clear_btn.click(
                        fn=clear_chat,
                        outputs=[chatbot]
                    )
                    
                    visit_websites_btn.click(
                        fn=self.visit_websites,
                        inputs=[website_urls],
                        outputs=[website_status]
                    )
    
    def visit_websites(self, urls_text):
        """Visit websites and extract content for context"""
        if not urls_text or not urls_text.strip():
            return " Please enter at least one URL"
        
        try:
            # Parse URLs
            urls = [url.strip() for url in urls_text.split(',') if url.strip()]
            if not urls:
                return " No valid URLs found"
            
            # Validate URLs
            valid_urls = []
            for url in urls:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                valid_urls.append(url)
            
            if not self.app_manager or not self.app_manager.web_manager:
                return " Web manager not available"
            
            # Visit websites and extract content
            visited_count = 0
            failed_count = 0
            total_content_length = 0
            
            for url in valid_urls:
                try:
                    result = asyncio.run(self.app_manager.web_manager.scrape_url(url))
                    if result:
                        visited_count += 1
                        total_content_length += len(result.get('content', ''))
                    
                    # Store in session for use in chat
                    if not hasattr(self, 'website_context'):
                        self.website_context = []
                    
                    self.website_context.append({
                        'url': url,
                        'title': result.get('metadata', {}).get('title', 'Unknown'),
                        'content': result.get('content', ''),
                        'timestamp': time.time()
                    })
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to visit {url}: {e}")
            
            # Clean old website context (keep only last 10)
            if hasattr(self, 'website_context') and len(self.website_context) > 10:
                self.website_context = self.website_context[-10:]
            
            if visited_count > 0:
                return f" Successfully visited {visited_count} website(s)\n Total content: {total_content_length:,} characters\n Failed: {failed_count}"
            else:
                return f" Failed to visit any websites. Errors: {failed_count}"
        
        except Exception as e:
            return f" Error visiting websites: {str(e)}"
    
    def _create_model_tab(self):
        """Create model management interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Model Management")
                
                # Model discovery
                with gr.Group():
                    gr.Markdown("### Model Discovery")
                    
                    model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="/path/to/models",
                        info="Path to scan for models"
                    )
                    
                    with gr.Row():
                        scan_btn = gr.Button("Scan Models", variant="primary")
                        refresh_btn = gr.Button("Refresh", variant="secondary")
                    
                    model_list = gr.Dataframe(
                        headers=["Model Name", "Path", "Size", "Status"],
                        datatype=["str", "str", "str", "str"],
                        label="Available Models",
                        interactive=False
                    )
                
                # Hugging Face Download
                with gr.Group():
                    gr.Markdown("### Hugging Face Download")
                    
                    hf_model_id = gr.Textbox(
                        label="Model ID",
                        placeholder="microsoft/DialoGPT-medium",
                        info="Hugging Face model identifier (e.g., microsoft/DialoGPT-medium)"
                    )
                    
                    with gr.Row():
                        hf_download_path = gr.Textbox(
                            label="Download Path",
                            placeholder="./models/",
                            value="./models/",
                            info="Local directory to download the model"
                        )
                    
                    hf_use_auth_token = gr.Checkbox(
                        value=False,
                        label="Use Auth Token",
                        info="Use HF_TOKEN environment variable for private models"
                    )
                    
                    with gr.Accordion("Download Options", open=False):
                        hf_revision = gr.Textbox(
                            label="Revision",
                            placeholder="main",
                            value="main",
                            info="Git revision (branch, tag, or commit hash)"
                        )
                        
                        hf_cache_dir = gr.Textbox(
                            label="Cache Directory",
                            placeholder="~/.cache/huggingface/hub",
                            info="Local cache directory (optional)"
                        )
                        
                        hf_force_download = gr.Checkbox(
                            value=False,
                            label="Force Download",
                            info="Re-download even if model exists locally"
                        )
                    
                    with gr.Row():
                        hf_download_btn = gr.Button("Download Model", variant="primary")
                        hf_check_btn = gr.Button("Check Model", variant="secondary")
                    
                    hf_download_status = gr.Markdown("Ready to download models from Hugging Face")
                
                # Model loading
                with gr.Group():
                    gr.Markdown("### Model Loading")
                    
                    selected_model = gr.Dropdown(
                        label="Select Model",
                        choices=self._get_available_models(),
                        value=None,
                        interactive=True
                    )
                    
                    # Model Configuration
                    with gr.Accordion("Model Configuration", open=False):
                        context_length = gr.Slider(
                            minimum=1024,
                            maximum=262144,
                            value=32768,
                            step=1024,
                            label="Context Length",
                            info="Maximum sequence length (affects memory usage)"
                        )
                        
                        gpu_memory_util = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.9,
                            step=0.05,
                            label="GPU Memory Utilization",
                            info="Fraction of GPU memory to use"
                        )
                    
                    # Advanced GPU Controls
                    with gr.Accordion("Advanced GPU Settings", open=False):
                        with gr.Row():
                            tensor_parallel_size = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=1,
                                step=1,
                                label="Tensor Parallel Size",
                                info="Number of GPUs for tensor parallelism"
                            )
                            
                            pipeline_parallel_size = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="Pipeline Parallel Size",
                                info="Number of pipeline stages"
                            )
                        
                        with gr.Row():
                            gpu_selection = gr.CheckboxGroup(
                                choices=[],  # Will be populated dynamically
                                label="Select GPUs",
                                info="Choose specific GPUs to use (leave empty for auto)"
                            )
                            
                            refresh_gpu_list_btn = gr.Button("Refresh GPU List", size="sm")
                        
                        # Mixed GPU Setup Information
                        gr.Markdown(
                            """
**Mixed GPU Setup Detected**: 
- For multiple GPUs , use **Tensor Parallel Size = X** X=number of gpus to be utilized
- Consider using **CPU Offload** for KV cache instead (will experience slowdown)
- Set **CUDA_VISIBLE_DEVICES** environment variable to select specific GPU
""",
                            elem_classes=["info-text"]
                        )
                    
                    kv_cache_dtype = gr.Dropdown(
                        choices=["auto", "fp16", "fp8", "int8"],
                        value="auto",
                        label="KV Cache Data Type",
                        info="Data type for KV cache (affects memory usage)"
                    )
                    
                    cpu_offload_gb = gr.Slider(
                        minimum=0,
                        maximum=64,
                        value=0,
                        step=1,
                        label="CPU Offload (GB)",
                        info="Amount of KV cache to offload to CPU memory"
                    )
                    
                    quantization_method = gr.Dropdown(
                        choices=["none", "gptq", "awq", "autoround", "int4", "int8", "fp8"],
                        value="none",
                        label="Quantization",
                        info="Quantization method for model compression"
                    )
                    
                    enforce_eager = gr.Checkbox(
                        value=False,
                        label="Enforce Eager Mode",
                        info="Disable CUDA graphs for debugging"
                    )
                    
                    with gr.Row():
                        load_model_btn = gr.Button("Load Model", variant="primary")
                        unload_model_btn = gr.Button("Unload Model", variant="secondary")
                    
                    model_status_display = gr.Markdown("No model loaded")
                
                # Right column - GPU and system info
                with gr.Column():
                    gr.Markdown("## System Information")
                    
                    # GPU information
                    with gr.Group():
                        gr.Markdown("### GPU Status")
                        
                        gpu_info = gr.Dataframe(
                            headers=["GPU", "Name", "Memory", "Utilization", "Temperature"],
                            datatype=["str", "str", "str", "str", "str"],
                            label="GPU Information",
                            interactive=False
                        )
                        
                        refresh_gpu_btn = gr.Button("Refresh GPU Info", variant="secondary")
                    
                    # Event handlers
                    def scan_models(path):
                        try:
                            import asyncio
                            import threading
                            
                            # Check if we're in an event loop
                            try:
                                loop = asyncio.get_running_loop()
                                # We're in an event loop, use a thread
                                result = None
                                exception = None
                                
                                def run_async():
                                    nonlocal result, exception
                                    try:
                                        new_loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(new_loop)
                                        result = new_loop.run_until_complete(self.app_manager.scan_models(path))
                                        new_loop.close()
                                    except Exception as e:
                                        exception = e
                                
                                thread = threading.Thread(target=run_async)
                                thread.start()
                                thread.join()
                                
                                if exception:
                                    raise exception
                                
                            except RuntimeError:
                                # No event loop running, safe to use asyncio.run
                                result = asyncio.run(self.app_manager.scan_models(path))
                            
                            if result['success']:
                                models_data = []
                                model_choices = []
                                for model in result['models']:
                                    models_data.append([
                                        model['name'],
                                        model['path'],
                                        f"{model.get('size_gb', 0):.1f} GB",
                                        "Current" if model.get('is_current', False) else "Available"
                                    ])
                                    model_choices.append(model['name'])
                                
                                # Return both table data and dropdown choices
                                return models_data, gr.Dropdown(choices=model_choices, value=model_choices[0] if model_choices else None)
                            else:
                                error_data = [["Error", result.get('error', 'Unknown error'), "", ""]]
                                return error_data, gr.Dropdown(choices=[], value=None)
                        except Exception as e:
                            error_data = [["Error", str(e), "", ""]]
                            return error_data, gr.Dropdown(choices=[], value=None)
                    
                    def refresh_model_dropdown():
                        """Refresh the model dropdown with available models"""
                        try:
                            model_choices = self._get_available_models()
                            return gr.Dropdown(choices=model_choices, value=model_choices[0] if model_choices else None)
                        except Exception as e:
                            return gr.Dropdown(choices=[], value=None)
                    
                    def refresh_gpu_list():
                        """Refresh the GPU selection list"""
                        try:
                            if self.app_manager and self.app_manager.vllm_manager:
                                gpu_info = self.app_manager.vllm_manager.get_gpu_status()
                                gpu_choices = [f"GPU {gpu['id']}: {gpu['name']}" for gpu in gpu_info]
                                return gr.CheckboxGroup(choices=gpu_choices)
                            else:
                                return gr.CheckboxGroup(choices=[])
                        except Exception as e:
                            return gr.CheckboxGroup(choices=[])
                    
                    def load_selected_model(model_name, context_len, gpu_mem_util, tensor_parallel, pipeline_parallel, selected_gpus, kv_dtype, cpu_offload, quant_method, eager_mode):
                        try:
                            # Parse selected GPUs
                            gpu_ids = []
                            if selected_gpus:
                                for gpu_str in selected_gpus:
                                    # Extract GPU ID from "GPU 0: NVIDIA RTX 4090" format
                                    gpu_id = int(gpu_str.split(':')[0].replace('GPU ', ''))
                                    gpu_ids.append(gpu_id)
                            
                            config = {
                                'max_model_len': int(context_len),
                                'context_length': int(context_len),
                                'gpu_memory_utilization': float(gpu_mem_util),
                                'tensor_parallel_size': int(tensor_parallel),
                                'pipeline_parallel_size': int(pipeline_parallel),
                                'gpu_ids': gpu_ids if gpu_ids else None,
                                'kv_cache_dtype': kv_dtype,
                                'cpu_offload_gb': int(cpu_offload),
                                'quantization': quant_method if quant_method != 'none' else None,
                                'enforce_eager': bool(eager_mode)
                            }
                            
                            result = asyncio.run(self.app_manager.load_model(model_name, config))
                            if result['success']:
                                gpu_info = f"GPUs: {gpu_ids}" if gpu_ids else "GPUs: Auto"
                                return f" Model loaded successfully: {model_name}\n Context: {context_len:,} tokens\n GPU Memory: {gpu_mem_util:.1%}\n Tensor Parallel: {tensor_parallel}\n Pipeline Parallel: {pipeline_parallel}\n {gpu_info}\n KV Cache: {kv_dtype}\n CPU Offload: {cpu_offload}GB"
                            else:
                                return f" Failed to load model: {result.get('error', 'Unknown error')}"
                        except Exception as e:
                            return f" Error: {str(e)}"
                    
                    def unload_current_model():
                        try:
                            result = asyncio.run(self.app_manager.unload_model())
                            if result['success']:
                                return " Model unloaded successfully"
                            else:
                                return f" Failed to unload model: {result.get('error', 'Unknown error')}"
                        except Exception as e:
                            return f" Error: {str(e)}"
                    
                    def get_gpu_status():
                        try:
                            result = asyncio.run(self.app_manager.get_gpu_status())
                            if result['success']:
                                gpu_data = []
                                for gpu in result['gpus']:
                                    gpu_data.append([
                                        f"GPU {gpu['id']}",
                                        gpu['memory_used'],
                                        gpu['memory_total'],
                                        f"{gpu['utilization']}%"
                                    ])
                                return gpu_data
                            else:
                                return [["Error", result.get('error', 'Unknown error'), "", ""]]
                        except Exception as e:
                            return [["Error", str(e), "", ""]]
                    
                    def download_hf_model(model_id, download_path, use_auth_token, revision, cache_dir, force_download):
                        try:
                            if not model_id.strip():
                                return " Please enter a model ID"
                            
                            import os
                            from huggingface_hub import snapshot_download
                            
                            # Prepare download arguments
                            download_args = {
                                'repo_id': model_id.strip(),
                                'local_dir': os.path.join(download_path, model_id.split('/')[-1]),
                                'revision': revision if revision.strip() else 'main',
                                'force_download': force_download
                            }
                            
                            # Add auth token if requested
                            if use_auth_token:
                                token = os.getenv('HF_TOKEN')
                                if token:
                                    download_args['token'] = token
                                else:
                                    return " HF_TOKEN environment variable not found"
                            
                            # Add cache directory if specified
                            if cache_dir.strip():
                                download_args['cache_dir'] = cache_dir.strip()
                            
                            # Create download directory if it doesn't exist
                            os.makedirs(download_args['local_dir'], exist_ok=True)
                            
                            # Download the model
                            result_path = snapshot_download(**download_args)
                            
                            return f" Model downloaded successfully!\n Path: {result_path}\n You can now scan for models to load it."
                        
                        except ImportError:
                            return " huggingface_hub not installed. Run: pip install huggingface_hub"
                        except Exception as e:
                            return f" Download failed: {str(e)}"
                    
                    def check_hf_model(model_id, use_auth_token):
                        try:
                            if not model_id.strip():
                                return " Please enter a model ID"
                            
                            from huggingface_hub import HfApi
                            
                            api = HfApi()
                            
                            # Prepare API arguments
                            api_args = {'repo_id': model_id.strip()}
                            
                            # Add auth token if requested
                            if use_auth_token:
                                token = os.getenv('HF_TOKEN')
                                if token:
                                    api_args['token'] = token
                                else:
                                    return " HF_TOKEN environment variable not found"
                            
                            # Get model info
                            model_info = api.model_info(**api_args)
                            
                            # Format model information
                            info_text = f" Model found: {model_info.modelId}\n"
                            info_text += f" Downloads: {model_info.downloads:,}\n"
                            info_text += f" Likes: {model_info.likes:,}\n"
                            info_text += f" Last Modified: {model_info.lastModified}\n"
                            
                            if model_info.pipeline_tag:
                                info_text += f" Pipeline: {model_info.pipeline_tag}\n"
                            
                            if hasattr(model_info, 'library_name') and model_info.library_name:
                                info_text += f" Library: {model_info.library_name}\n"
                            
                            # Get file sizes
                            try:
                                files = api.list_repo_files(model_info.modelId)
                                model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
                                if model_files:
                                    info_text += f" Model Files: {len(model_files)} files\n"
                            except:
                                pass
                            
                            return info_text
                        
                        except ImportError:
                            return " huggingface_hub not installed. Run: pip install huggingface_hub"
                        except Exception as e:
                            return f" Model check failed: {str(e)}"
                    
                    # Connect events
                    scan_btn.click(
                        fn=scan_models,
                        inputs=[model_path],
                        outputs=[model_list, selected_model]
                    )
                    
                    refresh_btn.click(
                        fn=refresh_model_dropdown,
                        outputs=[selected_model]
                    )
                    
                    load_model_btn.click(
                        fn=load_selected_model,
                        inputs=[selected_model, context_length, gpu_memory_util, tensor_parallel_size, pipeline_parallel_size, gpu_selection, kv_cache_dtype, cpu_offload_gb, quantization_method, enforce_eager],
                        outputs=[model_status_display]
                    )
                    
                    refresh_gpu_list_btn.click(
                        fn=refresh_gpu_list,
                        outputs=[gpu_selection]
                    )
                    
                    unload_model_btn.click(
                        fn=unload_current_model,
                        outputs=[model_status_display]
                    )
                    
                    hf_download_btn.click(
                        fn=download_hf_model,
                        inputs=[hf_model_id, hf_download_path, hf_use_auth_token, hf_revision, hf_cache_dir, hf_force_download],
                        outputs=[hf_download_status]
                    )
                    
                    hf_check_btn.click(
                        fn=check_hf_model,
                        inputs=[hf_model_id, hf_use_auth_token],
                        outputs=[hf_download_status]
                    )
                    
                    refresh_gpu_btn.click(
                        fn=get_gpu_status,
                        outputs=[gpu_info]
                    )
    
    def _create_rag_tab(self):
        """Create RAG management interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## RAG Management")
                
                # Document ingestion
                with gr.Group():
                    gr.Markdown("### Document Ingestion")
                    
                    with gr.Tabs():
                        with gr.Tab("Text Input"):
                            text_input = gr.Textbox(
                                label="Text Content",
                                lines=10,
                                placeholder="Paste your text content here..."
                            )
                            
                            text_metadata = gr.Textbox(
                                label="Metadata (JSON)",
                                placeholder='{"source": "manual_input", "title": "My Document"}',
                                lines=2
                            )
                            
                            ingest_text_btn = gr.Button(" Ingest Text", variant="primary")
                        
                        with gr.Tab("File Upload"):
                            file_upload = gr.File(
                                label="Upload Documents",
                                file_count="multiple",
                                file_types=[".txt", ".pdf", ".docx", ".md"]
                            )
                            
                            ingest_files_btn = gr.Button("Ingest Files", variant="primary")
                        
                        with gr.Tab("Web URLs"):
                            url_input = gr.Textbox(
                                label="URLs (one per line)",
                                lines=5,
                                placeholder="https://example.com/article1\nhttps://example.com/article2"
                            )
                            
                            ingest_urls_btn = gr.Button("Ingest URLs", variant="primary")
                
                # RAG status and controls
                with gr.Group():
                    gr.Markdown("### RAG Status")
                    
                    rag_stats = gr.JSON(
                        label="RAG Statistics",
                        value={}
                    )
                    
                    with gr.Row():
                        refresh_rag_btn = gr.Button(" Refresh Stats")
                        clear_rag_btn = gr.Button("Clear All Documents", variant="stop")
                
                # Document search and retrieval
                with gr.Group():
                    gr.Markdown("### Document Search")
                    
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search query..."
                    )
                    
                    search_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                    
                    search_btn = gr.Button(" Search Documents", variant="primary")
                    
                    search_results = gr.Dataframe(
                        headers=["Content", "Source", "Score"],
                        datatype=["str", "str", "number"],
                        label="Search Results",
                        interactive=False
                    )
                
                # Event handlers
                def ingest_text_content(text, metadata_str):
                    try:
                        if not text.strip():
                            return " Please provide text content"
                        
                        metadata = {}
                        if metadata_str.strip():
                            metadata = json.loads(metadata_str)
                        
                        result = asyncio.run(self.app_manager.ingest_text([text], [metadata]))
                        if result['success']:
                            return f" Text ingested successfully. {result['documents_processed']} documents processed."
                        else:
                            return f" Failed to ingest text: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                def ingest_uploaded_files(files):
                    try:
                        if not files:
                            return " Please upload files"
                        
                        file_paths = [file.name for file in files]
                        result = asyncio.run(self.app_manager.ingest_files(file_paths))
                        
                        if result['success']:
                            return f" Files ingested successfully. {result['documents_processed']} documents processed."
                        else:
                            return f" Failed to ingest files: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                def ingest_web_urls(urls_text):
                    try:
                        if not urls_text.strip():
                            return " Please provide URLs"
                        
                        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                        result = asyncio.run(self.app_manager.ingest_urls(urls))
                        
                        if result['success']:
                            return f" URLs ingested successfully. {result['documents_processed']} documents processed."
                        else:
                            return f" Failed to ingest URLs: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                def get_rag_statistics():
                    try:
                        result = asyncio.run(self.app_manager.get_rag_stats())
                        return result
                    except Exception as e:
                        return {"error": str(e)}
                
                def search_documents(query, k):
                    try:
                        if not query.strip():
                            return []
                        
                        result = asyncio.run(self.app_manager.search_documents(query, k))
                        if result['success']:
                            search_data = []
                            for doc in result['documents']:
                                search_data.append([
                                    doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                                    doc.get('source', 'Unknown'),
                                    round(doc.get('score', 0), 3)
                                ])
                            return search_data
                        else:
                            return [["Error", result.get('error', 'Unknown error'), 0]]
                    except Exception as e:
                        return [["Error", str(e), 0]]
                
                def clear_all_documents():
                    try:
                        result = asyncio.run(self.app_manager.clear_rag_documents())
                        if result['success']:
                            return " All documents cleared successfully"
                        else:
                            return f" Failed to clear documents: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                # Connect events
                ingest_text_btn.click(
                    fn=ingest_text_content,
                    inputs=[text_input, text_metadata],
                    outputs=[gr.Textbox(label="Status", interactive=False)]
                )
                
                ingest_files_btn.click(
                    fn=ingest_uploaded_files,
                    inputs=[file_upload],
                    outputs=[gr.Textbox(label="Status", interactive=False)]
                )
                
                ingest_urls_btn.click(
                    fn=ingest_web_urls,
                    inputs=[url_input],
                    outputs=[gr.Textbox(label="Status", interactive=False)]
                )
                
                refresh_rag_btn.click(
                    fn=get_rag_statistics,
                    outputs=[rag_stats]
                )
                
                search_btn.click(
                    fn=search_documents,
                    inputs=[search_query, search_k],
                    outputs=[search_results]
                )
                
                clear_rag_btn.click(
                    fn=clear_all_documents,
                    outputs=[gr.Textbox(label="Status", interactive=False)]
                )
    
    def _create_files_tab(self):
        """Create file management interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## File Management")
                
                # File upload
                with gr.Group():
                    gr.Markdown("### File Upload")
                    
                    file_uploader = gr.File(
                        label="Upload Files",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".docx", ".doc", ".jpg", ".png", ".csv", ".json"]
                    )
                    
                    upload_btn = gr.Button(" Upload Files", variant="primary")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                # File list
                with gr.Group():
                    gr.Markdown("### Uploaded Files")
                    
                    files_list = gr.Dataframe(
                        headers=["File ID", "Filename", "Size", "Type", "Status"],
                        datatype=["str", "str", "str", "str", "str"],
                        label="Files",
                        interactive=False
                    )
                
                with gr.Row():
                    refresh_files_btn = gr.Button("Refresh")
                    process_all_btn = gr.Button("Process All")
                    delete_selected_btn = gr.Button("Delete Selected", variant="stop")
                
                # File details
                with gr.Group():
                    gr.Markdown("### File Details")
                    
                    file_id_input = gr.Textbox(
                        label="File ID",
                        placeholder="Enter file ID to view details"
                    )
                    
                    view_file_btn = gr.Button("View File")
                    
                    file_content = gr.Textbox(
                        label="File Content",
                        lines=10,
                        interactive=False
                    )
                    
                    file_metadata = gr.JSON(
                        label="File Metadata",
                        value={}
                    )
                
                # Event handlers
                def upload_files(files):
                    try:
                        if not files:
                            return " No files selected"
                        
                        results = []
                        for file in files:
                            result = asyncio.run(self.app_manager.upload_file(file.name))
                            results.append(result)
                        
                        successful = sum(1 for r in results if r['success'])
                        total = len(results)
                        
                        return f" Uploaded {successful}/{total} files successfully"
                    
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                def get_files_list():
                    try:
                        result = asyncio.run(self.app_manager.list_files())
                        if result['success']:
                            files_data = []
                            for file_info in result['files']:
                                files_data.append([
                                    file_info['file_id'],
                                    file_info['filename'],
                                    f"{file_info['size']} bytes",
                                    file_info.get('file_type', 'Unknown'),
                                    "Processed" if file_info['processed'] else "Pending"
                                ])
                            return files_data
                        else:
                            return [["Error", result.get('error', 'Unknown error'), "", "", ""]]
                    except Exception as e:
                        return [["Error", str(e), "", "", ""]]
                
                def view_file_details(file_id):
                    try:
                        if not file_id.strip():
                            return "", {}
                        
                        result = asyncio.run(self.app_manager.get_file_info(file_id))
                        if result['success']:
                            file_info = result['file_info']
                            content = file_info.get('content', 'No content available')
                            metadata = file_info.get('metadata', {})
                            
                            return content, metadata
                        else:
                            return f"Error: {result.get('error', 'Unknown error')}", {}
                    except Exception as e:
                        return f"Error: {str(e)}", {}
                
                def process_all_files():
                    try:
                        result = asyncio.run(self.app_manager.process_all_files())
                        if result['success']:
                            return f" Processed {result['processed_count']}/{result['total_count']} files"
                        else:
                            return f" Error: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                # Connect events
                upload_btn.click(
                    fn=upload_files,
                    inputs=[file_uploader],
                    outputs=[upload_status]
                )
                
                refresh_files_btn.click(
                    fn=get_files_list,
                    outputs=[files_list]
                )
                
                view_file_btn.click(
                    fn=view_file_details,
                    inputs=[file_id_input],
                    outputs=[file_content, file_metadata]
                )
                
                process_all_btn.click(
                    fn=process_all_files,
                    outputs=[upload_status]
                )
    
    def _create_web_tab(self):
        """Create web functionality interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Web Integration")
                
                # Web search
                with gr.Group():
                    gr.Markdown("### Web Search")
                    
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search query..."
                    )
                    
                    with gr.Row():
                        search_provider_select = gr.Dropdown(
                            label="Search Provider",
                            choices=["auto", "duckduckgo", "google", "bing"],
                            value="auto"
                        )
                        
                        max_results_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Max Results"
                        )
                    
                    web_search_btn = gr.Button(" Search Web", variant="primary")
                    
                    search_results_display = gr.Dataframe(
                        headers=["Title", "URL", "Snippet"],
                        datatype=["str", "str", "str"],
                        label="Search Results",
                        interactive=False
                    )
                
                # URL scraping
                with gr.Group():
                    gr.Markdown("### URL Scraping")
                    
                    urls_input = gr.Textbox(
                        label="URLs (one per line)",
                        lines=5,
                        placeholder="https://example.com/page1\nhttps://example.com/page2"
                    )
                    
                    scrape_btn = gr.Button("Scrape URLs", variant="primary")
                    
                    scrape_results = gr.Dataframe(
                        headers=["URL", "Title", "Content Length", "Status"],
                        datatype=["str", "str", "str", "str"],
                        label="Scraping Results",
                        interactive=False
                    )
                
                # Web enhancement
                with gr.Group():
                    gr.Markdown("### Web-Enhanced Generation")
                    
                    enhancement_query = gr.Textbox(
                        label="Query",
                        placeholder="Enter query for web-enhanced response..."
                    )
                    
                    with gr.Row():
                        include_search_check = gr.Checkbox(
                            label="Include Search",
                            value=True
                        )
                        
                        include_scraping_check = gr.Checkbox(
                            label="Include Scraping",
                            value=True
                        )
                    
                    enhance_btn = gr.Button(" Enhance with Web", variant="primary")
                    
                    enhancement_results = gr.Textbox(
                        label="Enhancement Results",
                        lines=10,
                        interactive=False
                    )
                
                # Event handlers
                def search_web(query, provider, max_results):
                    try:
                        if not query.strip():
                            return []
                        
                        result = asyncio.run(self.app_manager.search_web(
                            query, max_results, provider if provider != "auto" else None
                        ))
                        
                        if result['success']:
                            search_data = []
                            for item in result['results']:
                                search_data.append([
                                    item['title'],
                                    item['url'],
                                    item['snippet'][:200] + "..." if len(item['snippet']) > 200 else item['snippet']
                                ])
                            return search_data
                        else:
                            return [["Error", result.get('error', 'Unknown error'), ""]]
                    except Exception as e:
                        return [["Error", str(e), ""]]
                
                def scrape_urls(urls_text):
                    try:
                        if not urls_text.strip():
                            return []
                        
                        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                        result = asyncio.run(self.app_manager.scrape_urls(urls))
                        
                        if result['success']:
                            scrape_data = []
                            for item in result['results']:
                                scrape_data.append([
                                    item['url'],
                                    item.get('title', 'No title'),
                                    f"{len(item.get('content', ''))} chars",
                                    "Success" if item.get('success', False) else "Failed"
                                ])
                            return scrape_data
                        else:
                            return [["Error", result.get('error', 'Unknown error'), "", ""]]
                    except Exception as e:
                        return [["Error", str(e), "", ""]]
                
                def enhance_with_web(query, include_search, include_scraping):
                    try:
                        if not query.strip():
                            return " Please provide a query"
                        
                        result = asyncio.run(self.app_manager.enhance_with_web(
                            query, include_search, include_scraping
                        ))
                        
                        if result['success']:
                            return f" Web enhancement completed.\n\nDocuments found: {len(result['documents'])}\nProcessing time: {result['processing_time']:.2f}s"
                        else:
                            return f" Enhancement failed: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                # Connect events
                web_search_btn.click(
                    fn=search_web,
                    inputs=[search_input, search_provider_select, max_results_slider],
                    outputs=[search_results_display]
                )
                
                scrape_btn.click(
                    fn=scrape_urls,
                    inputs=[urls_input],
                    outputs=[scrape_results]
                )
                
                enhance_btn.click(
                    fn=enhance_with_web,
                    inputs=[enhancement_query, include_search_check, include_scraping_check],
                    outputs=[enhancement_results]
                )
    
    def _create_settings_tab(self):
        """Create settings interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Settings")
                
                # Model settings
                with gr.Group():
                    gr.Markdown("### Model Settings")
                    
                    model_cache_dir = gr.Textbox(
                        label="Model Cache Directory",
                        value="./models",
                        info="Directory to cache downloaded models"
                    )
                    
                    default_model = gr.Textbox(
                        label="Default Model",
                        placeholder="model-name",
                        info="Default model to load on startup"
                    )
                
                # RAG settings
                with gr.Group():
                    gr.Markdown("### RAG Settings")
                    
                    chunk_size = gr.Number(
                        label="Chunk Size",
                        value=1000,
                        minimum=100,
                        maximum=5000,
                        step=100,
                        info="Size of text chunks for RAG"
                    )
                    
                    chunk_overlap = gr.Number(
                        label="Chunk Overlap",
                        value=200,
                        minimum=0,
                        maximum=1000,
                        step=50,
                        info="Overlap between chunks"
                    )
                    
                    embedding_model = gr.Textbox(
                        label="Embedding Model",
                        value="sentence-transformers/all-MiniLM-L6-v2",
                        info="Model for generating embeddings"
                    )
                
                # Web settings
                with gr.Group():
                    gr.Markdown("### Web Settings")
                    
                    web_timeout = gr.Number(
                        label="Web Timeout (seconds)",
                        value=30,
                        minimum=5,
                        maximum=120,
                        step=5,
                        info="Timeout for web requests"
                    )
                    
                    max_web_content = gr.Number(
                        label="Max Web Content Length",
                        value=100000,
                        minimum=10000,
                        maximum=1000000,
                        step=10000,
                        info="Maximum length of web content to process"
                    )
                
                # Save settings
                save_settings_btn = gr.Button(" Save Settings", variant="primary")
                settings_status = gr.Textbox(label="Status", interactive=False)
                
                # Event handlers
                def save_settings(model_cache, default_mod, chunk_sz, chunk_ovlp, 
                                embed_model, web_to, max_web):
                    try:
                        settings = {
                            'model_cache_dir': model_cache,
                            'default_model': default_mod,
                            'chunk_size': chunk_sz,
                            'chunk_overlap': chunk_ovlp,
                            'embedding_model': embed_model,
                            'web_timeout': web_to,
                            'max_web_content': max_web
                        }
                        
                        result = asyncio.run(self.app_manager.save_settings(settings))
                        if result['success']:
                            return " Settings saved successfully"
                        else:
                            return f" Failed to save settings: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f" Error: {str(e)}"
                
                # Connect events
                save_settings_btn.click(
                    fn=save_settings,
                    inputs=[
                        model_cache_dir, default_model, chunk_size, chunk_overlap,
                        embedding_model, web_timeout, max_web_content
                    ],
                    outputs=[settings_status]
                )
    
    def _create_system_tab(self):
        """Create system monitoring interface"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## System Status")
                
                # System information
                with gr.Group():
                    gr.Markdown("### System Information")
                    
                    system_info = gr.JSON(
                        label="System Info",
                        value={}
                    )
                    
                    refresh_system_btn = gr.Button("Refresh System Info")
                
                # Performance metrics
                with gr.Group():
                    gr.Markdown("### Performance Metrics")
                    
                    performance_metrics = gr.Dataframe(
                        headers=["Metric", "Value", "Unit"],
                        datatype=["str", "str", "str"],
                        label="Metrics",
                        interactive=False
                    )
                    
                    refresh_metrics_btn = gr.Button("Refresh Metrics")
                
                # Logs
                with gr.Group():
                    gr.Markdown("### Recent Logs")
                    
                    log_level = gr.Dropdown(
                        label="Log Level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        value="INFO"
                    )
                    
                    log_lines = gr.Number(
                        label="Number of Lines",
                        value=50,
                        minimum=10,
                        maximum=500,
                        step=10
                    )
                    
                    get_logs_btn = gr.Button(" Get Logs")
                    
                    logs_display = gr.Textbox(
                        label="Logs",
                        lines=15,
                        interactive=False
                    )
                
                # Health check
                with gr.Group():
                    gr.Markdown("### Health Check")
                    
                    health_status = gr.JSON(
                        label="Health Status",
                        value={}
                    )
                    
                    health_check_btn = gr.Button(" Run Health Check", variant="primary")
                
                # System controls
                with gr.Group():
                    gr.Markdown("### System Controls")
                    
                    system_control_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="System running normally"
                    )
                    
                    with gr.Row():
                        restart_ui_btn = gr.Button("Restart UI", variant="secondary")
                        shutdown_btn = gr.Button("Shutdown", variant="stop")
                
                # Event handlers
                def get_system_info():
                    try:
                        result = asyncio.run(self.app_manager.get_system_info())
                        return result
                    except Exception as e:
                        return {"error": str(e)}
                
                def get_performance_metrics():
                    try:
                        result = asyncio.run(self.app_manager.get_performance_metrics())
                        if result['success']:
                            metrics_data = []
                            for metric, value in result['metrics'].items():
                                unit = result.get('units', {}).get(metric, '')
                                metrics_data.append([metric, str(value), unit])
                            return metrics_data
                        else:
                            return [["Error", result.get('error', 'Unknown error'), ""]]
                    except Exception as e:
                        return [["Error", str(e), ""]]
                
                def get_recent_logs(level, lines):
                    try:
                        result = asyncio.run(self.app_manager.get_logs(level, lines))
                        if result['success']:
                            return result['logs']
                        else:
                            return f"Error: {result.get('error', 'Unknown error')}"
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                def run_health_check():
                    try:
                        result = asyncio.run(self.app_manager.health_check())
                        return result
                    except Exception as e:
                        return {"error": str(e)}
                
                def restart_ui():
                    """Restart the UI by restarting main.py"""
                    try:
                        import subprocess
                        import sys
                        import os
                        
                        # Get the path to main.py
                        main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'main.py')
                        
                        # Restart the application
                        subprocess.Popen([sys.executable, main_path])
                        
                        # Exit current process after a short delay
                        import time
                        time.sleep(1)
                        os._exit(0)
                        
                        return " Restarting UI..."
                    except Exception as e:
                        return f" Failed to restart: {str(e)}"
                
                def shutdown_system():
                    """Shutdown the system"""
                    try:
                        import os
                        import time
                        
                        # Give time for the response to be sent
                        time.sleep(0.5)
                        
                        # Exit the application
                        os._exit(0)
                        
                        return " Shutting down..."
                    except Exception as e:
                        return f" Failed to shutdown: {str(e)}"
                
                # Connect events
                refresh_system_btn.click(
                    fn=get_system_info,
                    outputs=[system_info]
                )
                
                refresh_metrics_btn.click(
                    fn=get_performance_metrics,
                    outputs=[performance_metrics]
                )
                
                get_logs_btn.click(
                    fn=get_recent_logs,
                    inputs=[log_level, log_lines],
                    outputs=[logs_display]
                )
                
                health_check_btn.click(
                    fn=run_health_check,
                    outputs=[health_status]
                )
                
                restart_ui_btn.click(
                    fn=restart_ui,
                    outputs=[system_control_status]
                )
                
                shutdown_btn.click(
                    fn=shutdown_system,
                    outputs=[system_control_status]
                )
    
    def _get_context_usage_display(self) -> str:
        """Get formatted context usage display"""
        percentage = (self.context_tokens_used / self.max_context_length * 100) if self.max_context_length > 0 else 0
        return f" Context: {self.context_tokens_used:,} / {self.max_context_length:,} tokens ({percentage:.1f}%)"
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            import asyncio
            import threading
            
            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use a thread
                result = None
                exception = None
                
                def run_async():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(self.app_manager.get_available_models())
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                result = asyncio.run(self.app_manager.get_available_models())
            
            if result['success']:
                return [model['name'] for model in result['models']]
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if not self.interface:
            self.interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': False,
            'show_error': True,
            'quiet': False
        }
        
        # Update with provided parameters
        launch_params.update(kwargs)
        
        logger.info(f"Launching Gradio interface on {launch_params['server_name']}:{launch_params['server_port']}")
        
        return self.interface.launch(**launch_params)
