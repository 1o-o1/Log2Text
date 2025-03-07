import os
import torch
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# JSON schema for authentication analysis (shortened reference for prompt)
JSON_SCHEMA_SHORT = '''{
  "log_type": "Authentication",
  "field_descriptions": {...},
  "observations": {
    "source_actor": "...",
    "targets": {...},
    "temporal_patterns": {...},
    "behavioral_patterns": {...}
  },
  "potential_indicators": {...},
  "next_steps_for_validation": {...},
  "conclusion": {...},
  "high_risk_indicators":     { "required": [
          "anonymous_logon_detected",
          "unknown_auth_type",
          "ntlm_in_kerberos_env",
          "machine_account_anomalies",
          "multiple_accounts_single_source",
          "lateral_movement_indicators",
          "excessive_ticket_requests",
          "incomplete_session_pairs"
        ],}
}'''

def get_target_modules_for_model(model_name):
    """
    Get appropriate LoRA target modules based on model architecture.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        list: List of target module names
    """
    if "Qwen/QwQ" in model_name:
        # Target modules for QwQ models
        return ["c_attn", "c_proj", "w1", "w2"]
    elif "DeepSeek" in model_name and "Llama" in model_name:
        # Target modules for DeepSeek-Llama models
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        # Default target modules for most LLaMA-like models
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

def create_config(data_dir="./data", output_dir="./models"):
    """
    Create configuration dictionary optimized for low memory 8GB GPU.
    
    Args:
        data_dir (str): Data directory path
        output_dir (str): Output directory path
        
    Returns:
        dict: Configuration dictionary
    """
    config = {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "data_dir": data_dir,
        "output_dir": output_dir,
        "sft_model_path": os.path.join(output_dir, "sft_model"),
        "reward_model_path": os.path.join(output_dir, "reward_model"),
        "rl_model_path": os.path.join(output_dir, "rl_model"),
        
        # LoRA settings
        "lora_r": 4,  # Very small rank for memory efficiency
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        
        # Training settings
        "max_seq_length": 1024,  # Reduced for memory efficiency
        "batch_size": 1,  # Single example batches
        "gradient_accumulation_steps": 16,  # Accumulate gradients
        "num_train_epochs": 1,  # Reduced epochs
        "learning_rate": 2e-5,
        "warmup_steps": 10,
        "max_train_samples": 100,  # Limit training samples
        "ppo_epochs": 1,
        "seed": 42,
        
        # Memory optimization
        "use_4bit": True,  # Use 4-bit quantization
        "use_cpu_offload": True,  # Offload to CPU
        "use_gradient_checkpointing": True,
        "clear_cache_every_n_steps": 5,
    }
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config["sft_model_path"], exist_ok=True)
    os.makedirs(config["reward_model_path"], exist_ok=True)
    os.makedirs(config["rl_model_path"], exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set random seed
    torch.manual_seed(config["seed"])
    
    return config

# Full JSON schema for reference (load only when needed)
def load_full_schema():
    """Load the full JSON schema from file or embedded"""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            return f.read()
    else:
        # Embedded schema for fallback
        return """{ 
  "log_type": "Authentication",
  "field_descriptions": {
    "source_computer": "Computer initiating the authentication",
    "destination_computer": "Target computer for authentication",
    "auth_type": "Authentication protocol used",
    "logon_type": "Type of logon",
    "times": "Timestamp(s) of authentication events",
    "source_users": "User account(s) originating the authentication",
    "destination_users": "User account(s) targeted for authentication",
    "orientations": "Authentication operation (LogOn, LogOff, TGS, TGT)",
    "statuses": "Outcome of authentication attempt (Success, Failure)"
  },
  "observations": {
    "source_actor": "Analysis of auth initiators",
    "targets": {
      "frequent_targets": ["system1", "system2"],
      "sporadic_targets": ["system3"]
    },
    "temporal_patterns": {
      "clusters": "Time periods with concentrated activity",
      "bursts": "Sudden spikes in authentication volume",
      "off_hours_activity": "Authentication outside business hours"
    },
    "behavioral_patterns": {
      "repetitive_actions": "Recurring patterns",
      "lateral_movement": "Sequential patterns between systems",
      "privilege_escalation": "Patterns suggesting privilege increase"
    }
  },
  "potential_indicators": {
    "suspicious_auth_types": {
      "description": "Analysis of suspicious types",
      "affected_entities": ["system1", "system2"]
    },
    "account_patterns": {
      "description": "Suspicious account usage",
      "affected_accounts": ["account1"]
    },
    "logon_logoff_sequences": {
      "description": "Suspicious sequences",
      "affected_entities": ["system1"]
    },
    "anomalies": {
      "description": "Overall anomalies",
      "deviation_details": "Details about deviations"
    }
  },
  "next_steps_for_validation": {
    "temporal_correlation": "Steps to analyze timing",
    "behavioral_context": "Methods to compare with baselines",
    "permission_analysis": "Process to verify permissions",
    "ioc_checks": "Techniques to check"
  },
  "conclusion": {
    "summary": "Overall assessment",
    "recommended_actions": "Recommendations"
  },
  "high_risk_indicators": {
    "anonymous_logon_detected": true,
    "unknown_auth_type": true,
    "ntlm_in_kerberos_env": false,
    "machine_account_anomalies": false,
    "multiple_accounts_single_source": false,
    "lateral_movement_indicators": false,
    "excessive_ticket_requests": false,
    "incomplete_session_pairs": false
  }
}"""
