import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import Adafactor, default_data_collator
from model_utils import load_base_model, add_lora_adapter, clear_gpu_memory

logger = logging.getLogger(__name__)

class SFTTrainer:
    """
    Trainer for supervised fine-tuning (SFT) with LoRA.
    Optimized for memory efficiency on GPUs with limited memory.
    """
    
    def __init__(self, config, tokenizer, data_processor):
        """
        Initialize SFT trainer.
        
        Args:
            config (dict): Configuration dictionary
            tokenizer: Tokenizer
            data_processor: Data processor object
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.output_dir = config["sft_model_path"]
    
    def train(self):
        """
        Run SFT training with custom loop for memory efficiency.
        
        Returns:
            str: Path to saved model
        """
        # Check if model already exists
        if os.path.exists(os.path.join(self.output_dir, "adapter_model.bin")):
            logger.info(f"SFT model already exists at {self.output_dir}. Skipping training.")
            return self.output_dir
        
        logger.info("Starting SFT training...")
        
        # Prepare dataset
        dataset = self.data_processor.prepare_sft_dataset(
            max_samples=self.config.get("max_train_samples", 100)
        )
        
        # Load base model
        model = load_base_model(self.config, model_type="causal_lm")
        
        # Add LoRA adapter
        model = add_lora_adapter(model, self.config)
        
        # Use custom training loop instead of Trainer
        self._custom_training_loop(model, dataset)
        
        # Save model
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"SFT model saved to {self.output_dir}")
        
        # Clean up memory
        del model
        clear_gpu_memory()
        
        return self.output_dir
    
    def _custom_training_loop(self, model, dataset):
        """
        Custom training loop optimized for memory efficiency.
        
        Args:
            model: Model to train
            dataset: Dataset to train on
        """
        logger.info("Starting custom training loop...")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=default_data_collator
        )
        
        # Use Adafactor optimizer for memory efficiency
        optimizer = Adafactor(
            model.parameters(),
            lr=self.config["learning_rate"],
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
        
        # Training parameters
        num_epochs = self.config["num_train_epochs"]
        accumulation_steps = self.config["gradient_accumulation_steps"]
        clear_cache_steps = self.config.get("clear_cache_every_n_steps", 10)
        
        # Main training loop
        model.train()
        for epoch in range(int(num_epochs)):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            steps = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to appropriate device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Log step loss
                epoch_loss += loss.item() * accumulation_steps
                steps += 1
                
                # Update weights
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Log progress
                    if steps % 10 == 0:
                        avg_loss = epoch_loss / steps
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {steps}, Loss: {avg_loss:.4f}")
                
                # Clear CUDA cache periodically
                if (batch_idx + 1) % clear_cache_steps == 0:
                    clear_gpu_memory()
            
            # Make sure to update weights at the end of epoch if needed
            if steps % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log epoch summary
            avg_epoch_loss = epoch_loss / steps
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Clear cache at the end of epoch
            clear_gpu_memory()
