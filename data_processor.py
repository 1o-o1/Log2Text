import os
import json
import logging
import pandas as pd
from datasets import Dataset
from config import JSON_SCHEMA_SHORT

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles all data loading, preprocessing, and dataset creation.
    Optimized for memory efficiency.
    """
    
    def __init__(self, config, tokenizer):
        """
        Initialize data processor.
        
        Args:
            config (dict): Configuration dictionary 
            tokenizer: Tokenizer for the model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.auth_folder = os.path.join(config["data_dir"], "auth_data", "benign")
        self.json_folder = os.path.join(config["data_dir"], "Auth_bin")
    
    def load_and_preprocess_data(self, max_samples=None):
        """
        Load and preprocess training data.
        
        Args:
            max_samples (int, optional): Maximum number of samples to load
            
        Returns:
            tuple: (sft_dataset, pref_dataset) - Datasets for SFT and preference learning
        """
        logger.info("Loading and preprocessing data...")
        
        # Get list of auth data files
        auth_files = []
        for file in os.listdir(self.auth_folder):
            if file.startswith("auth_data_") and file.endswith(".txt"):
                auth_files.append(os.path.join(self.auth_folder, file))
        
        # Limit number of files if specified
        if max_samples is not None and max_samples < len(auth_files):
            auth_files = auth_files[:max_samples]
            logger.info(f"Limited to {max_samples} samples")
        
        # Match auth data with annotations
        matched_data = []
        for auth_file in auth_files:
            file_id = os.path.basename(auth_file).split('_')[2]
            time_stamp = os.path.basename(auth_file).split('_')[3].split('.')[0]
            
            # Find corresponding JSON file
            json_file_name = f"auth_text_{file_id}_{time_stamp}.json"
            json_file_path = os.path.join(self.json_folder, json_file_name)
            
            if os.path.exists(json_file_path):
                # Load data from files
                with open(auth_file, 'r') as f:
                    auth_data = f.read()
                
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Create prompt with simplified schema to save memory
                prompt = f"""Task: Analyze the following authentication data and generate a security analysis in JSON format.

Authentication Data:
{auth_data}

Generate a detailed JSON analysis using this format:
{JSON_SCHEMA_SHORT}"""

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
    
    def prepare_sft_dataset(self, max_samples=None):
        """
        Prepare dataset for supervised fine-tuning.
        
        Args:
            max_samples (int, optional): Maximum number of samples to include
            
        Returns:
            Dataset: Tokenized dataset ready for training
        """
        # Load raw datasets
        sft_dataset, _ = self.load_and_preprocess_data(
            max_samples=max_samples or self.config.get("max_train_samples")
        )
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config["max_seq_length"],
                return_tensors=None  # Don't convert to tensors yet
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = sft_dataset.map(tokenize_function, batched=True, desc="Tokenizing")
        
        # Add labels for causal language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        logger.info("Adding labels...")
        tokenized_dataset = tokenized_dataset.map(add_labels, desc="Adding labels")
        
        return tokenized_dataset
    
    def prepare_preference_dataset(self, max_samples=None):
        """
        Prepare dataset for preference learning.
        
        Args:
            max_samples (int, optional): Maximum number of samples to include
            
        Returns:
            Dataset: Dataset with chosen/rejected pairs
        """
        # Load raw datasets
        _, pref_dataset = self.load_and_preprocess_data(
            max_samples=max_samples or self.config.get("max_train_samples")
        )
        
        # Limit to specified number of samples
        if max_samples is not None and max_samples < len(pref_dataset):
            pref_dataset = pref_dataset.select(range(max_samples))
            
        return pref_dataset
