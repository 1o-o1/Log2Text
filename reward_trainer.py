import os
import json
import torch
import logging
import shutil
from model_utils import load_base_model, add_lora_adapter, clear_gpu_memory
from peft import TaskType

logger = logging.getLogger(__name__)

class RewardModelTrainer:
    """
    Trainer for creating and training reward models.
    Includes a direct training method and an alternative approach for very memory-constrained setups.
    """
    
    def __init__(self, config, tokenizer, data_processor):
        """
        Initialize reward model trainer.
        
        Args:
            config (dict): Configuration dictionary
            tokenizer: Tokenizer
            data_processor: Data processor
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.output_dir = config["reward_model_path"]
    
    def train(self):
        """
        Train a reward model or create a simplified version based on available memory.
        
        Returns:
            str: Path to reward model
        """
        # Check if model already exists
        if os.path.exists(os.path.join(self.output_dir, "adapter_model.bin")):
            logger.info(f"Reward model already exists at {self.output_dir}. Skipping training.")
            return self.output_dir
        
        # Try to detect if we have enough memory for full training
        has_enough_memory = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 7 * 1024**3
        
        if has_enough_memory:
            logger.info("Attempting full reward model training...")
            try:
                return self._train_reward_model()
            except RuntimeError as e:
                # If we run out of memory, fall back to alternative
                if "CUDA out of memory" in str(e):
                    logger.warning("Ran out of CUDA memory. Falling back to alternative approach.")
                    clear_gpu_memory()
                    return self._create_alternative_reward_model()
                else:
                    raise
        else:
            logger.info("Not enough GPU memory detected. Using alternative reward model approach.")
            return self._create_alternative_reward_model()
    
    def _train_reward_model(self):
        """
        Train a full reward model using preference data.
        
        Returns:
            str: Path to trained model
        """
        logger.info("Training reward model with preference learning...")
        
        # Prepare dataset
        pref_dataset = self.data_processor.prepare_preference_dataset(
            max_samples=min(50, self.config.get("max_train_samples", 100))
        )
        
        # Extract data
        chosen_texts = pref_dataset["chosen"]
        rejected_texts = pref_dataset["rejected"]
        
        # Load classification model
        model = load_base_model(self.config, model_type="seq_cls", num_labels=1)
        
        # Add LoRA
        model = add_lora_adapter(model, self.config, task_type=TaskType.SEQ_CLS)
        
        # Move model to device
        device = next(model.parameters()).device
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=0.01
        )
        
        # Batch processing
        chunk_size = 4
        num_chunks = (len(chosen_texts) + chunk_size - 1) // chunk_size
        grad_acc_steps = 4
        
        # Training loop
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
                    max_length=min(2048, self.config["max_seq_length"]),  # Use even smaller context
                    return_tensors="pt"
                )
                
                rejected_inputs = self.tokenizer(
                    rejected_chunk, 
                    padding="max_length",
                    truncation=True,
                    max_length=min(2048, self.config["max_seq_length"]),
                    return_tensors="pt"
                )
                
                # Process each item in chunk individually
                for i in range(len(chosen_chunk)):
                    # Get tensors
                    chosen_input_ids = chosen_inputs["input_ids"][i:i+1].to(device)
                    chosen_attention_mask = chosen_inputs["attention_mask"][i:i+1].to(device)
                    rejected_input_ids = rejected_inputs["input_ids"][i:i+1].to(device)
                    rejected_attention_mask = rejected_inputs["attention_mask"][i:i+1].to(device)
                    
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
                    
                    # Bradley-Terry loss
                    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards))
                    loss = loss.mean() / grad_acc_steps
                    
                    # Backward
                    loss.backward()
                    
                    epoch_loss += loss.item() * grad_acc_steps
                    num_batches += 1
                    
                    # Step
                    if num_batches % grad_acc_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Free memory
                    del chosen_input_ids, chosen_attention_mask, rejected_input_ids
                    del rejected_attention_mask, chosen_outputs, rejected_outputs
                    del chosen_rewards, rejected_rewards, loss
                
                # Log progress and clear cache
                logger.info(f"Epoch {epoch+1}, Chunk {chunk_idx+1}/{num_chunks} processed")
                clear_gpu_memory()
            
            # Final step if needed
            if num_batches % grad_acc_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log epoch results
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{self.config['num_train_epochs']}, Avg Loss: {avg_loss:.4f}")
        
        # Save model
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Reward model saved to {self.output_dir}")
        
        # Clean up
        del model
        clear_gpu_memory()
        
        return self.output_dir
    
    def _create_alternative_reward_model(self):
        """
        Create a simplified reward model using metadata and simple heuristics.
        This is used when memory constraints prevent full training.
        
        Returns:
            str: Path to the simplified reward model
        """
        logger.info("Creating simplified reward model (metadata only)...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a minimal model configuration
        config_data = {
            "model_type": "synthetic_reward",
            "base_model": self.config["model_name"],
            "reward_signals": {
                "positive": [
                    "suspicious",
                    "anomalous",
                    "unauthorized",
                    "risk",
                    "alert",
                    "malicious"
                ],
                "negative": [
                    "benign",
                    "normal",
                    "legitimate",
                    "expected",
                    "routine"
                ]
            },
            "version": "0.1",
            "method": "heuristic"
        }
        
        # Save configuration
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Save tokenizer config
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Copy SFT adapter if it exists to simulate a starting point
        sft_path = self.config["sft_model_path"]
        if os.path.exists(os.path.join(sft_path, "adapter_model.bin")):
            logger.info("Copying SFT adapter to reward model as starting point")
            for filename in ["adapter_model.bin", "adapter_config.json"]:
                src_file = os.path.join(sft_path, filename)
                if os.path.exists(src_file):
                    dst_file = os.path.join(self.output_dir, filename)
                    shutil.copy(src_file, dst_file)
        
        logger.info(f"Simplified reward model created at {self.output_dir}")
        return self.output_dir
