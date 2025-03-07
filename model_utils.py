import os
import gc
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)
from config import get_target_modules_for_model

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_tokenizer(model_name_or_path):
    """
    Load tokenizer with proper padding token.
    
    Args:
        model_name_or_path (str): Model name or path
        
    Returns:
        tokenizer: Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def get_bits_and_bytes_config(use_4bit=True, cpu_offload=True):
    """
    Create a BitsAndBytesConfig object for quantization.
    
    Args:
        use_4bit (bool): Whether to use 4-bit quantization
        cpu_offload (bool): Whether to enable CPU offloading
        
    Returns:
        BitsAndBytesConfig: Quantization configuration
    """
    if not use_4bit:
        return None
        
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Using float16 instead of bfloat16 for compatibility
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
        llm_int8_skip_modules=["lm_head"]  # Skip some modules from quantization
    )

def get_device_map(gpu_layers=4):
    """
    Create a device map that places only a few layers on GPU.
    
    Args:
        gpu_layers (int): Number of layers to put on GPU
        
    Returns:
        dict: Device mapping
    """
    # Basic device map
    device_map = {
        "model.embed_tokens": 0,
        "lm_head": "cpu"  # Put lm_head on CPU to save GPU memory
    }
    
    # Add layers
    num_layers = 32  # Typical number of layers in an 8B model
    
    for i in range(num_layers):
        if i < gpu_layers:
            device_map[f"model.layers.{i}"] = 0
        else:
            device_map[f"model.layers.{i}"] = "cpu"
    
    device_map["model.norm"] = 0  # Final layer norm on GPU
    
    return device_map

def load_base_model(config, model_type="causal_lm", num_labels=None):
    """
    Load base model with memory optimizations.
    
    Args:
        config (dict): Configuration dictionary
        model_type (str): Type of model to load ("causal_lm" or "seq_cls")
        num_labels (int, optional): Number of labels for classification
        
    Returns:
        model: Loaded model
    """
    # Clear GPU memory first
    clear_gpu_memory()
    
    model_name = config["model_name"]
    logger.info(f"Loading {model_type} model from {model_name}")
    
    # Setup quantization config
    bnb_config = get_bits_and_bytes_config(
        use_4bit=config.get("use_4bit", True),
        cpu_offload=config.get("use_cpu_offload", True)
    )
    
    # Try different loading strategies
    try:
        # First attempt: Load with auto device mapping
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:  # seq_cls
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels or 1,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
    except Exception as e:
        logger.warning(f"First loading attempt failed: {str(e)}")
        clear_gpu_memory()
        
        # Second attempt: Try with manual device mapping
        device_map = get_device_map(gpu_layers=3)  # Only 3 layers on GPU
        
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:  # seq_cls
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels or 1,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
    
    # Prepare model for k-bit training if quantization is used
    if config.get("use_4bit", True):
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if specified
    if config.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    
    return model

def add_lora_adapter(model, config, task_type=TaskType.CAUSAL_LM):
    """
    Add LoRA adapter to model.
    
    Args:
        model: Base model
        config (dict): Configuration dictionary
        task_type: TaskType for the LoRA adapter
        
    Returns:
        model: Model with LoRA adapter
    """
    logger.info(f"Adding LoRA adapter with rank {config['lora_r']}")
    
    # Get target modules for the model
    target_modules = get_target_modules_for_model(config["model_name"])
    
    # For sequence classification models, ensure the score/classifier layer is properly handled
    if task_type == TaskType.SEQ_CLS:
        # Convert the classification head to float32 to ensure proper gradient flow
        if hasattr(model, "score"):
            model.score.weight = torch.nn.Parameter(model.score.weight.float())
        elif hasattr(model, "classifier"):
            if hasattr(model.classifier, "weight"):
                model.classifier.weight = torch.nn.Parameter(model.classifier.weight.float())
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
        modules_to_save=[] if task_type == TaskType.SEQ_CLS else None  # Don't include classifier in modules_to_save
    )
    
    # Apply LoRA
    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except RuntimeError as e:
        if "only Tensors of floating point dtype can require gradients" in str(e):
            logger.warning("Issue with non-floating point tensors. Falling back to alternative approach.")
            # In this case, we'll skip certain problematic layers
            new_target_modules = [module for module in target_modules if module != "score" and module != "classifier"]
            peft_config.target_modules = new_target_modules
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            raise
    
    return model


def load_adapter_model(base_model_name_or_path, adapter_path, device="auto"):
    """
    Load a model with LoRA adapter.
    
    Args:
        base_model_name_or_path (str): Base model name or path
        adapter_path (str): Path to adapter
        device (str): Device to load model on
        
    Returns:
        model: Model with adapter
    """
    logger.info(f"Loading adapter model from {adapter_path}")
    
    # Clear memory first
    clear_gpu_memory()
    
    try:
        # Load quantized base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
    except Exception as e:
        logger.error(f"Error loading adapter model: {str(e)}")
        logger.info("Falling back to base model without adapter...")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map=device,
            trust_remote_code=True
        )
    
    return model
