import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    default_data_collator
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
json_schema = '''{
  "name": "authentication_log_analysis",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "log_type": {
        "type": "string",
        "description": "The type of log being analyzed (Authentication)",
        "enum": ["Authentication"]
      },
      "field_descriptions": {
        "type": "object",
        "properties": {
          "source_computer": {
            "type": "string",
            "description": "Computer initiating the authentication"
          },
          "destination_computer": {
            "type": "string",
            "description": "Target computer for authentication"
          },
          "auth_type": {
            "type": "string",
            "description": "Authentication protocol used (Kerberos, NTLM, etc.)"
          },
          "logon_type": {
            "type": "string",
            "description": "Type of logon (Network, Interactive, etc.)"
          },
          "times": {
            "type": "string",
            "description": "Timestamp(s) of authentication events"
          },
          "source_users": {
            "type": "string",
            "description": "User account(s) originating the authentication"
          },
          "destination_users": {
            "type": "string",
            "description": "User account(s) targeted for authentication"
          },
          "orientations": {
            "type": "string",
            "description": "Authentication operation (LogOn, LogOff, TGS, TGT)"
          },
          "statuses": {
            "type": "string",
            "description": "Outcome of authentication attempt (Success, Failure)"
          }
        },
        "required": [
          "source_computer",
          "destination_computer",
          "auth_type",
          "logon_type",
          "times",
          "source_users",
          "destination_users",
          "orientations",
          "statuses"
        ],
        "additionalProperties": false
      },
      "observations": {
        "type": "object",
        "properties": {
          "source_actor": {
            "type": "string",
            "description": "Analysis of computers and accounts initiating authentication events, noting suspicious naming patterns or unexpected systems"
          },
          "targets": {
            "type": "object",
            "properties": {
              "frequent_targets": {
                "type": "array",
                "description": "Frequently accessed systems, especially domain controllers or critical infrastructure",
                "items": {
                  "type": "string"
                }
              },
              "sporadic_targets": {
                "type": "array",
                "description": "Rarely accessed systems showing unusual authentication patterns",
                "items": {
                  "type": "string"
                }
              }
            },
            "required": [
              "frequent_targets",
              "sporadic_targets"
            ],
            "additionalProperties": false
          },
          "temporal_patterns": {
            "type": "object",
            "properties": {
              "clusters": {
                "type": "string",
                "description": "Time periods with concentrated authentication activity"
              },
              "bursts": {
                "type": "string",
                "description": "Sudden spikes in authentication volume"
              },
              "off_hours_activity": {
                "type": "string",
                "description": "Authentication events occurring outside business hours"
              }
            },
            "required": [
              "clusters",
              "bursts",
              "off_hours_activity"
            ],
            "additionalProperties": false
          },
          "behavioral_patterns": {
            "type": "object",
            "properties": {
              "repetitive_actions": {
                "type": "string",
                "description": "Recurring authentication patterns between systems"
              },
              "lateral_movement": {
                "type": "string",
                "description": "Sequential authentication patterns suggesting movement between systems"
              },
              "privilege_escalation": {
                "type": "string",
                "description": "Authentication patterns indicating escalation to higher privilege accounts"
              }
            },
            "required": [
              "repetitive_actions",
              "lateral_movement",
              "privilege_escalation"
            ],
            "additionalProperties": false
          }
        },
        "required": [
          "source_actor",
          "targets",
          "temporal_patterns",
          "behavioral_patterns"
        ],
        "additionalProperties": false
      },
      "potential_indicators": {
        "type": "object",
        "properties": {
          "suspicious_auth_types": {
            "type": "object",
            "properties": {
              "description": {
                "type": "string",
                "description": "Analysis of suspicious authentication types detected (ANONYMOUS LOGON, NTLM, unknown types)"
              },
              "affected_entities": {
                "type": "array",
                "description": "List of systems and accounts using suspicious authentication methods",
                "items": {
                  "type": "string"
                }
              }
            },
            "required": [
              "description",
              "affected_entities"
            ],
            "additionalProperties": false
          },
          "account_patterns": {
            "type": "object",
            "properties": {
              "description": {
                "type": "string",
                "description": "Analysis of suspicious account usage patterns"
              },
              "affected_accounts": {
                "type": "array",
                "description": "List of accounts showing suspicious authentication behavior",
                "items": {
                  "type": "string"
                }
              }
            },
            "required": [
              "description",
              "affected_accounts"
            ],
            "additionalProperties": false
          },
          "logon_logoff_sequences": {
            "type": "object",
            "properties": {
              "description": {
                "type": "string",
                "description": "Analysis of suspicious logon/logoff sequences or TGS/TGT request patterns"
              },
              "affected_entities": {
                "type": "array",
                "description": "List of systems and accounts showing irregular authentication sequences",
                "items": {
                  "type": "string"
                }
              }
            },
            "required": [
              "description",
              "affected_entities"
            ],
            "additionalProperties": false
          },
          "anomalies": {
            "type": "object",
            "properties": {
              "description": {
                "type": "string",
                "description": "Overall analysis of authentication anomalies detected"
              },
              "deviation_details": {
                "type": "string",
                "description": "Specific details about deviations from normal authentication patterns"
              }
            },
            "required": [
              "description",
              "deviation_details"
            ],
            "additionalProperties": false
          }
        },
        "required": [
          "suspicious_auth_types",
          "account_patterns",
          "logon_logoff_sequences",
          "anomalies"
        ],
        "additionalProperties": false
      },
      "next_steps_for_validation": {
        "type": "object",
        "properties": {
          "temporal_correlation": {
            "type": "string",
            "description": "Steps to analyze the sequence and timing of authentication events to identify attack chains"
          },
          "behavioral_context": {
            "type": "string",
            "description": "Methods to compare observed authentication patterns with known baselines"
          },
          "permission_analysis": {
            "type": "string",
            "description": "Process to verify whether authenticated accounts should have legitimate access to target systems"
          },
          "ioc_checks": {
            "type": "string",
            "description": "Specific techniques to check (Pass-the-Hash, Kerberoasting, etc.) based on observed authentication patterns"
          }
        },
        "required": [
          "temporal_correlation",
          "behavioral_context",
          "permission_analysis",
          "ioc_checks"
        ],
        "additionalProperties": false
      },
      "conclusion": {
        "type": "object",
        "properties": {
          "summary": {
            "type": "string",
            "description": "Summary assessment of authentication anomalies with risk classification"
          },
          "recommended_actions": {
            "type": "string",
            "description": "Specific recommendations for investigation and remediation of suspicious authentication events"
          }
        },
        "required": [
          "summary",
          "recommended_actions"
        ],
        "additionalProperties": false
      },
      "high_risk_indicators": {
        "type": "object",
        "properties": {
          "anonymous_logon_detected": {
            "type": "boolean",
            "description": "Indicates whether ANONYMOUS LOGON events were detected"
          },
          "unknown_auth_type": {
            "type": "boolean",
            "description": "Indicates whether authentication with missing/unknown (?) type was detected"
          },
          "ntlm_in_kerberos_env": {
            "type": "boolean",
            "description": "Indicates whether NTLM authentication was detected in a Kerberos-preferred environment"
          },
          "machine_account_anomalies": {
            "type": "boolean",
            "description": "Indicates whether machine accounts were authenticating to unusual systems"
          },
          "multiple_accounts_single_source": {
            "type": "boolean",
            "description": "Indicates whether multiple accounts were authenticating from a single source in a short timeframe"
          },
          "lateral_movement_indicators": {
            "type": "boolean",
            "description": "Indicates whether authentication chains suggest lateral movement"
          },
          "excessive_ticket_requests": {
            "type": "boolean",
            "description": "Indicates whether excessive TGS/TGT requests were detected"
          },
          "incomplete_session_pairs": {
            "type": "boolean",
            "description": "Indicates whether LogOn events without corresponding LogOff events were detected"
          }
        },
        "required": [
          "anonymous_logon_detected",
          "unknown_auth_type",
          "ntlm_in_kerberos_env",
          "machine_account_anomalies",
          "multiple_accounts_single_source",
          "lateral_movement_indicators",
          "excessive_ticket_requests",
          "incomplete_session_pairs"
        ],
        "additionalProperties": false
      }
    },
    "required": [
      "log_type",
      "field_descriptions",
      "observations",
      "potential_indicators",
      "next_steps_for_validation",
      "conclusion",
      "high_risk_indicators"
    ],
    "additionalProperties": false
  }
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

class DeepSeekGRPOTrainer:
    """
    Trainer for implementing GRPO (Generative Reinforcement from Preference Optimization)
    on authentication log analysis using DeepSeek models, optimized for limited hardware.
    """
    
    def __init__(self, config):
        """
        Initialize the trainer with configuration.
        
        Args:
            config (dict): Configuration dictionary containing model parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Get appropriate target modules for the model
        self.target_modules = get_target_modules_for_model(config["model_name"])
        logger.info(f"Using target modules for LoRA: {self.target_modules}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model paths
        self.sft_model_path = os.path.join(config["output_dir"], "sft_model")
        self.reward_model_path = os.path.join(config["output_dir"], "reward_model")
        self.rl_model_path = os.path.join(config["output_dir"], "rl_model")
        
        # Create output directories
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(self.sft_model_path, exist_ok=True)
        os.makedirs(self.reward_model_path, exist_ok=True)
        os.makedirs(self.rl_model_path, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
    
    def _load_and_preprocess_data(self):
        """
        Load and preprocess the training data.
        
        Returns:
            tuple: (sft_dataset, pref_dataset) - Datasets for SFT and preference learning
        """
        logger.info("Loading and preprocessing data...")
        
        # Load auth data files (input)
        auth_files = []
        auth_folder = os.path.join(self.config["data_dir"], "auth_data", "benign")
        for file in os.listdir(auth_folder):
            if file.startswith("auth_data_") and file.endswith(".txt"):
                auth_files.append(os.path.join(auth_folder, file))
        
        # Load annotation files (output/target)
        json_folder = os.path.join(self.config["data_dir"], "Auth_bin")
        json_files = []
        for file in os.listdir(json_folder):
            if file.endswith(".json"):
                json_files.append(os.path.join(json_folder, file))
        
        # Match auth data with annotations
        matched_data = []
        for auth_file in auth_files:
            file_id = os.path.basename(auth_file).split('_')[2]
            time_stamp = os.path.basename(auth_file).split('_')[3].split('.')[0]
            
            # Find corresponding JSON file
            json_file_name = f"auth_text_{file_id}_{time_stamp}.json"
            json_file_path = os.path.join(json_folder, json_file_name)
            
            if os.path.exists(json_file_path):
                with open(auth_file, 'r') as f:
                    auth_data = f.read()
                
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Create prompt and completion
                prompt = f"""Task: Analyze the following authentication data and generate a security analysis in JSON format.

                Authentication Data:
                {auth_data}

                Generate a detailed JSON analysis using following format
                Json: {json_schema}"""

                # Check if JSON data is a string or already parsed
                if isinstance(json_data, dict) and "analysis" in json_data:
                    completion = json_data["analysis"]
                else:
                    completion = json.dumps(json_data, indent=2)
                
                matched_data.append({
                    "prompt": prompt,
                    "completion": completion,
                    "auth_data": auth_data,
                    "json_data": json_data
                })
        
        logger.info(f"Loaded {len(matched_data)} matched data points")
        
        # Create dataset for supervised fine-tuning (SFT)
        sft_data = []
        for item in matched_data:
            # For simplicity, use a plain text format with clear separation
            text = f"{item['prompt']}\n\n{item['completion']}"
            sft_data.append({"text": text})
        
        # Create dataset for preference learning
        pref_data = []
        for item in matched_data:
            # Original completion is the "chosen" one
            chosen = f"{item['prompt']}\n\n{item['completion']}"
            
            # Create a corrupted version as the "rejected" one
            json_data = item["json_data"]
            if isinstance(json_data, dict):
                # Randomly modify some fields to create a worse version
                corrupted_json = json_data.copy()
                
                # Modify some fields to make it worse
                if "observations" in corrupted_json:
                    if "source_actor" in corrupted_json["observations"]:
                        corrupted_json["observations"]["source_actor"] = "No suspicious activity detected."
                
                if "conclusion" in corrupted_json:
                    if "summary" in corrupted_json["conclusion"]:
                        corrupted_json["conclusion"]["summary"] = "No significant security concerns identified."
                
                if "high_risk_indicators" in corrupted_json:
                    # Set all indicators to false
                    for key in corrupted_json["high_risk_indicators"]:
                        corrupted_json["high_risk_indicators"][key] = False
                
                rejected_completion = json.dumps(corrupted_json, indent=2)
            else:
                # If we can't parse the JSON, just use a generic rejection
                rejected_completion = "{\n  \"analysis\": \"No suspicious activity detected.\"\n}"
            
            rejected = f"{item['prompt']}\n\n{rejected_completion}"
            
            pref_data.append({
                "chosen": chosen,
                "rejected": rejected,
                "prompt": item["prompt"]
            })
        
        # Create datasets
        sft_dataset = Dataset.from_pandas(pd.DataFrame(sft_data))
        pref_dataset = Dataset.from_pandas(pd.DataFrame(pref_data))
        
        return sft_dataset, pref_dataset
    
    def _prepare_sft_model(self):
        """
        Prepare the model for supervised fine-tuning (SFT) with LoRA.
        
        Returns:
            model: The model with LoRA adapters
        """
        logger.info(f"Preparing SFT model for {self.config['model_name']}...")
        
        # Load the model with quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            #quantization_config=bnb_config,
            
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA with the appropriate target modules
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=self.target_modules
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def train_sft_model(self):
        """
        Perform supervised fine-tuning on the model.
        """
        logger.info("Starting supervised fine-tuning...")
        
        # Load and preprocess data
        sft_dataset, _ = self._load_and_preprocess_data()
        
        # Tokenize the data directly here (before creating the model)
        # This ensures we don't have memory issues during training
        def tokenize_function(examples):
            # Tokenize with padding and truncation
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config["max_seq_length"],
                return_tensors=None  # Important: Don't convert to tensors yet
            )
        
        # Process the entire dataset at once
        logger.info("Tokenizing dataset...")
        tokenized_dataset = sft_dataset.map(tokenize_function, batched=True, desc="Tokenizing")
        
        # Add labels for causal language modeling (copied from input_ids)
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        logger.info("Adding labels...")
        tokenized_dataset = tokenized_dataset.map(add_labels, desc="Adding labels")
        
        # Prepare model after tokenizing to save memory
        model = self._prepare_sft_model()
        
        # Training arguments with high gradient accumulation for memory efficiency
        training_args = TrainingArguments(
            output_dir=self.sft_model_path,
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=1,  # Very small batch size for low memory
            gradient_accumulation_steps=16,  # High accumulation to compensate
            warmup_steps=self.config["warmup_steps"],
            learning_rate=self.config["learning_rate"],
            fp16=torch.cuda.is_available(),
            logging_dir=os.path.join(self.config["output_dir"], "logs"),
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            remove_unused_columns=False,  # Important to keep the labels column
            optim="adamw_torch",
            # Memory optimization settings
            gradient_checkpointing=True,
            logging_first_step=True,
            max_grad_norm=1.0,
            # Enable memory savings
            torch_compile=False,  # Can enable if PyTorch 2.0+ available
            dataloader_drop_last=True
        )
        
        # Initialize trainer with default_data_collator instead of DataCollatorForLM
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=default_data_collator,  # Use the simpler default_data_collator
            tokenizer=self.tokenizer
        )
        
        # Train model
        logger.info("Starting training loop...")
        trainer.train()
        
        # Save model
        trainer.save_model(self.sft_model_path)
        logger.info(f"SFT model saved to {self.sft_model_path}")
        
        return self.sft_model_path
    
    def train_reward_model(self):
        """
        Train a reward model using preference data.
        """
        logger.info("Training reward model...")
        
        # Load and preprocess data
        _, pref_dataset = self._load_and_preprocess_data()
        
        # Load the model with quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=1,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA for reward model with the appropriate target modules
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=self.target_modules
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Custom dataloader with small batch size
        def custom_train_reward_model():
            # Process dataset in smaller chunks for memory efficiency
            batch_size = 1
            
            # Tokenize in batches to prevent OOM
            chosen_texts = pref_dataset["chosen"]
            rejected_texts = pref_dataset["rejected"]
            
            # Process in chunks
            chunk_size = 10
            num_chunks = (len(chosen_texts) + chunk_size - 1) // chunk_size
            
            # Move model to device
            model.to(self.device)
            model.train()
            
            # Create optimizer with weight decay to improve generalization
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=0.01
            )
            
            # Training loop with gradient accumulation
            grad_acc_steps = 8
            for epoch in range(self.config["num_train_epochs"]):
                epoch_loss = 0
                num_batches = 0
                optimizer.zero_grad()
                
                for chunk_idx in range(num_chunks):
                    # Get chunk indices
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(chosen_texts))
                    
                    # Process this chunk
                    chosen_chunk = chosen_texts[start_idx:end_idx]
                    rejected_chunk = rejected_texts[start_idx:end_idx]
                    
                    # Tokenize
                    chosen_inputs = self.tokenizer(
                        chosen_chunk,
                        padding="max_length",
                        truncation=True,
                        max_length=self.config["max_seq_length"],
                        return_tensors="pt"
                    )
                    
                    rejected_inputs = self.tokenizer(
                        rejected_chunk, 
                        padding="max_length",
                        truncation=True,
                        max_length=self.config["max_seq_length"],
                        return_tensors="pt"
                    )
                    
                    # Process each item in chunk separately
                    for i in range(len(chosen_chunk)):
                        # Get single item tensors
                        chosen_input_ids = chosen_inputs["input_ids"][i:i+1].to(self.device)
                        chosen_attention_mask = chosen_inputs["attention_mask"][i:i+1].to(self.device)
                        rejected_input_ids = rejected_inputs["input_ids"][i:i+1].to(self.device)
                        rejected_attention_mask = rejected_inputs["attention_mask"][i:i+1].to(self.device)
                        
                        # Forward pass for chosen
                        chosen_outputs = model(
                            input_ids=chosen_input_ids,
                            attention_mask=chosen_attention_mask
                        )
                        chosen_rewards = chosen_outputs.logits.squeeze()
                        
                        # Forward pass for rejected
                        rejected_outputs = model(
                            input_ids=rejected_input_ids,
                            attention_mask=rejected_attention_mask
                        )
                        rejected_rewards = rejected_outputs.logits.squeeze()
                        
                        # Calculate loss using Bradley-Terry preference model
                        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards))
                        loss = loss.mean() / grad_acc_steps  # Normalize by accumulation steps
                        
                        # Backward pass with gradient accumulation
                        loss.backward()
                        
                        epoch_loss += loss.item() * grad_acc_steps
                        num_batches += 1
                        
                        # Step every grad_acc_steps or at the end of an epoch
                        if num_batches % grad_acc_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    logger.info(f"Epoch {epoch+1}, Chunk {chunk_idx+1}/{num_chunks} processed")
                
                # Make sure to step if there are any remaining gradients
                if num_batches % grad_acc_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch+1}/{self.config['num_train_epochs']}, Avg Loss: {avg_loss:.4f}")
        
        # Train reward model
        custom_train_reward_model()
        
        # Save reward model
        model.save_pretrained(self.reward_model_path)
        self.tokenizer.save_pretrained(self.reward_model_path)
        logger.info(f"Reward model saved to {self.reward_model_path}")
        
        return self.reward_model_path
    
    def train_rl_model(self, sft_model_path=None, reward_model_path=None):
        """
        Train the model using RL from preferences (GRPO).
        
        Args:
            sft_model_path (str, optional): Path to SFT model. Defaults to None.
            reward_model_path (str, optional): Path to reward model. Defaults to None.
        """
        logger.info("Starting RL training with GRPO...")
        
        # Use provided paths or defaults
        sft_model_path = sft_model_path or self.sft_model_path
        reward_model_path = reward_model_path or self.reward_model_path
        
        # Load SFT model with value head for PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Load reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Load and preprocess data
        _, pref_dataset = self._load_and_preprocess_data()
        
        # Extract prompts for RL training - limit to 50 for memory constraints
        prompts = pref_dataset["prompt"][:50]
        
        # Configure PPO with memory-efficient settings
        ppo_config = PPOConfig(
            learning_rate=self.config["learning_rate"],
            batch_size=1,  # Very small batch for memory constraints
            mini_batch_size=1,
            gradient_accumulation_steps=8,
            optimize_device_cache=True,
            target_kl=0.2,  # Higher target KL to allow more exploration
            init_kl_coef=0.1,
            adap_kl_ctrl=True,
            ppo_epochs=self.config["ppo_epochs"],
            seed=self.config["seed"],
            use_score_scaling=True,  # Scale rewards for better stability
            use_score_norm=True
        )
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=self.tokenizer,
            dataset=prompts,
            data_collator=None
        )
        
        # Response length sampler - using smaller response length for memory
        response_length_sampler = LengthSampler(512, 768)
        
        # RL training loop
        for epoch in range(self.config["num_train_epochs"]):
            logger.info(f"Starting epoch {epoch+1}/{self.config['num_train_epochs']}")
            
            # Process smaller batches at a time
            dataloader_iter = iter(ppo_trainer.dataloader)
            num_batches = min(10, len(ppo_trainer.dataloader))  # Process at most 10 batches per epoch
            
            for batch_idx in range(num_batches):
                try:
                    batch = next(dataloader_iter)
                    batch_prompts = batch["input_ids"]
                    
                    # Generate responses one by one to save memory
                    response_tensors = []
                    for prompt_tensor in batch_prompts:
                        response_length = response_length_sampler()
                        
                        # Generate with more conservative parameters
                        response = ppo_trainer.generate(
                            prompt_tensor.unsqueeze(0),  # Add batch dimension
                            max_new_tokens=response_length,
                            do_sample=True,
                            temperature=0.8,
                            top_k=50,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        response_tensors.append(response.squeeze())
                        
                        # Clear CUDA cache to save memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Create full responses and compute rewards
                    texts = []
                    for i, response_tensor in enumerate(response_tensors):
                        # Use the decode/encode approach to save memory
                        prompt_text = self.tokenizer.decode(batch_prompts[i])
                        generated_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                        full_text = prompt_text + generated_text
                        texts.append(full_text)
                    
                    # Process reward computation in smaller chunks if needed
                    rewards = []
                    for text in texts:
                        # Tokenize individually
                        inputs = self.tokenizer(
                            text,
                            padding=True,
                            truncation=True,
                            max_length=self.config["max_seq_length"],
                            return_tensors="pt"
                        ).to(self.device)
                        
                        # Compute reward
                        with torch.no_grad():
                            output = reward_model(**inputs)
                            reward = output.logits.squeeze()
                            rewards.append(reward.item())
                    
                    # Convert to tensor
                    rewards = torch.tensor(rewards, device=self.device)
                    
                    # Update policy with PPO
                    stats = ppo_trainer.step(batch_prompts, response_tensors, rewards)
                    
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Stats: {stats}")
                    
                    # Clear CUDA cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error in PPO batch {batch_idx}: {str(e)}")
                    # Continue to next batch
                    continue
        
        # Save final RL model
        try:
            ppo_trainer.save_pretrained(self.rl_model_path)
            logger.info(f"RL model saved to {self.rl_model_path}")
        except Exception as e:
            logger.error(f"Error saving RL model: {str(e)}")
            # Try to save only the model weights
            torch.save(model.state_dict(), os.path.join(self.rl_model_path, "pytorch_model.bin"))
            self.tokenizer.save_pretrained(self.rl_model_path)
            logger.info(f"Saved RL model weights to {self.rl_model_path}")
        
        return self.rl_model_path
    
    def run_full_training_pipeline(self):
        """
        Run the complete GRPO training pipeline: SFT -> Reward Model -> RL
        """
        logger.info(f"Starting full GRPO training pipeline for {self.config['model_name']}")
        
        # Step 1: Supervised Fine-Tuning (SFT)
        sft_model_path = self.train_sft_model()
        
        # Step 2: Train Reward Model
        reward_model_path = self.train_reward_model()
        
        # Step 3: RL Fine-Tuning with GRPO
        rl_model_path = self.train_rl_model(sft_model_path, reward_model_path)
        
        logger.info("GRPO training pipeline completed successfully!")
        return rl_model_path
    
    def generate_analysis(self, auth_data, model_path=None):
        """
        Generate security analysis for authentication data using the trained model.
        
        Args:
            auth_data (str): Authentication data to analyze
            model_path (str, optional): Path to the model to use. Defaults to RL model.
            
        Returns:
            dict: JSON security analysis
        """
        # Use provided model path or default to RL model
        model_path = model_path or self.rl_model_path
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Create prompt
        prompt = f"""Task: Analyze the following authentication data and generate a security analysis in JSON format.

        Authentication Data:
        {auth_data}

        Generate a detailed JSON analysis using following format
                Json: {json_schema}"""
        
        # Tokenize prompt
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,  # Reduced for memory constraints
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract only the new tokens (exclude prompt tokens)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
        
        # Decode response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract JSON part from response
        try:
            # Find the start of the JSON content
            json_start = response.find("{")
            if json_start == -1:
                logger.error("Failed to find JSON content in model response")
                return {"error": "No JSON content found in response"}
            
            json_text = response[json_start:]
            analysis = json.loads(json_text)
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated JSON: {e}")
            return {"error": "Failed to parse JSON", "raw_response": response}


# Example usage
def main():
    # Configuration optimized for limited hardware (8GB GPU, 16GB RAM)
    config = {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "data_dir": "",
        "output_dir": "./models",
        "lora_r": 8,  # Reduced LoRA rank for memory efficiency
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "max_seq_length": 1024,  # Reduced sequence length for memory
        "batch_size": 1,  # Single sample batches
        "num_train_epochs": 2,  # Fewer epochs for faster training
        "learning_rate": 2e-5,
        "warmup_steps": 50,
        "ppo_epochs": 2,
        "seed": 42,
        "use_4bit": True  # Critical for 8GB GPU
    }
    
    # Initialize trainer
    trainer = DeepSeekGRPOTrainer(config)
    
    # Run full training pipeline
    trainer.run_full_training_pipeline()
    
    # Generate analysis for new data
    sample_auth_data = """Aggregated Authentication Events:
source_computer,destination_computer,auth_type,logon_type,times,source_users,destination_users,orientations,statuses
C11569,C2327,?,?,<150885>,U4757@DOM1,U4757@DOM1,TGS,Success
C457,C457,?,Network,<150885, 150885, 150885, 150885>,U501@DOM1,U6419@DOM1,U6577@DOM1,U7309@DOM1,U501@DOM1,U6419@DOM1,U6577@DOM1,U7309@DOM1,LogOff,Success
C586,C586,?,Network,<150885>,ANONYMOUS LOGON,ANONYMOUS LOGON,LogOn,Success
C467,C467,?,Network,<150885>,U5025@DOM1,U5025@DOM1,LogOff,Success"""
    
    analysis = trainer.generate_analysis(sample_auth_data)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()