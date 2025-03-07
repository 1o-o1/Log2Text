import torch
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def check_cuda_and_libraries():
    # Check PyTorch CUDA integration
    print("\n=== PyTorch CUDA Check ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Devices: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor on GPU
        x = torch.tensor([1.0, 2.0], device='cuda')
        print(f"Test Tensor on GPU: {x}")

    # Check bitsandbytes installation
    print("\n=== bitsandbytes Check ===")
    print(f"bitsandbytes Version: {bnb.__version__}")
    
    try:
        from bitsandbytes.nn import Linear8bitLt
        print("bitsandbytes Linear8bitLt import successful")
    except ImportError:
        print("bitsandbytes 8-bit linear layer import failed")

    # Check transformers integration with bitsandbytes
    print("\n=== Transformers Check ===")
    try:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("BitsAndBytesConfig created successfully")
    except Exception as e:
        print(f"Error creating BitsAndBytesConfig: {e}")

    # Check PEFT integration
    print("\n=== PEFT Check ===")
    try:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        print("LoRA config created successfully")
    except Exception as e:
        print(f"Error creating LoRA config: {e}")

    # Check model loading capability
    print("\n=== Model Loading Check ===")
    try:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Example model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully with quantization")
    except Exception as e:
        print(f"Error loading quantized model: {e}")

check_cuda_and_libraries()