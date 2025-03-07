import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import Adafactor
from model_utils import load_base_model, add_lora_adapter, clear_gpu_memory

logger = logging.getLogger(__name__)

class RLTrainer:
    """
    Trainer for reinforcement learning fine-tuning.
    Uses a simplified approach for memory efficiency instead of full PPO.
    """
    
    def __init__(self, config, tokenizer, data_processor):
        """
        Initialize RL trainer.
        
        Args:
            config (dict): Configuration dictionary
            tokenizer: Tokenizer
            data_processor: Data processor
        """
        self.config = config
        self.tokenizer = tokenizer
        self.data_processor = data_processor
        self.output_dir = config["rl_model_path"]
    
    def train(self, sft_model_path=None):
        """
        Run simplified RL fine-tuning.
        
        Args:
            sft_model_path (str, optional): Path to SFT model
            
        Returns:
            str: Path to RL model
        """
        # Check if model already exists
        if os.path.exists(os.path.join(self.output_dir, "adapter_model.bin")):
            logger.info(f"RL model already exists at {self.output_dir}. Skipping training.")
            return self.output_dir
            
        logger.info("Starting simplified RL training...")
        
        # Use specified SFT model path or default
        sft_path = sft_model_path or self.config["sft_model_path"]
        
        # Get a very small dataset of high-quality examples
        tokenized_dataset = self.data_processor.prepare_sft_dataset(
            max_samples=min(20, self.config.get("max_train_samples", 50))
        )
        
        # Try to load model with adapter
        try:
            logger.info(f"Loading SFT model from {sft_path}")
            from peft import PeftModel
            
            # Load base model first
            base_model = load_base_model(self.config, model_type="causal_lm")
            
            # Add adapter
            model = PeftModel.from_pretrained(base_model, sft_path)
            
        except Exception as e:
            logger.error(f"Error loading SFT model: {str(e)}")
            logger.info("Falling back to training from scratch")
            
            # Load fresh model and add adapter
            model = load_base_model(self.config, model_type="causal_lm")
            model = add_lora_adapter(model, self.config)
        
        # Run direct fine-tuning (simulating RL)
        self._direct_fine_tuning(model, tokenized_dataset)
        
        # Save model
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"RL model saved to {self.output_dir}")
        
        # Clean up
        del model
        clear_gpu_memory()
        
        return self.output_dir
    
    # def _direct_fine_tuning(self, model, dataset):
    #     """
    #     Simplified RL through direct fine-tuning on high-quality examples.
        
    #     Args:
    #         model: Model to fine-tune
    #         dataset: Dataset of high-quality examples
    #     """
    #     logger.info("Starting direct fine-tuning (simplified RL)...")
        
    #     # Create minimal dataloader
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=True
    #     )
        
    #     # Use Adafactor for memory efficiency
    #     optimizer = Adafactor(
    #         model.parameters(),
    #         lr=self.config["learning_rate"] / 2,  # Lower learning rate
    #         scale_parameter=False,
    #         relative_step=False,
    #         warmup_init=False
    #     )
        
    #     # Training parameters
    #     steps = 50  # Just do limited steps
    #     accumulation_steps = 4
        
    #     # Training loop
    #     model.train()
    #     step = 0
    #     total_loss = 0
        
    #     for batch_idx, batch in enumerate(dataloader):
    #         if step >= steps:
    #             break
                
    #         # Move to device
    #         batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
    #                  for k, v in batch.items()}
            
    #         # Forward pass
    #         outputs = model(**batch)
    #         loss = outputs.loss / accumulation_steps
            
    #         # Backward pass
    #         loss.backward()
            
    #         # Update
    #         if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
    #             optimizer.step()
    #             optimizer.zero_grad()
                
    #             # Log
    #             step += 1
    #             total_loss += loss.item() * accumulation_steps
                
    #             if step % 10 == 0:
    #                 avg_loss = total_loss / step
    #                 logger.info(f"Step {step}/{steps}, Loss: {avg_loss:.4f}")
                
    #             # Clear cache
    #             if step % 10 == 0:
    #                 clear_gpu_memory()
            
    #         # Break if we've done enough steps
    #         if step >= steps:
    #             break
        
    #     # Final logging
    #     avg_loss = total_loss / step if step > 0 else 0
    #     logger.info(f"Direct fine-tuning completed. Final loss: {avg_loss:.4f}")
    def _direct_fine_tuning(self, model, dataset):
        """
        Simplified RL through direct fine-tuning on high-quality examples.
        
        Args:
            model: Model to fine-tune
            dataset: Dataset of high-quality examples
        """
        logger.info("Starting direct fine-tuning (simplified RL)...")
        
        # Create minimal dataloader with proper collation
        from transformers import default_data_collator
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=default_data_collator  # Add proper collation function
        )
        
        # Use Adafactor for memory efficiency
        optimizer = Adafactor(
            model.parameters(),
            lr=self.config["learning_rate"] / 2,  # Lower learning rate
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
        
        # Training parameters
        steps = 50  # Just do limited steps
        accumulation_steps = 4
        
        # Training loop
        model.train()
        step = 0
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if step >= steps:
                break
                
            try:
                # Ensure all batch items are tensors and on the right device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=model.device) 
                        for k, v in batch.items()}
                
                # Print batch keys and shapes for debugging
                logger.debug(f"Batch keys: {batch.keys()}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"  {k}: {v.shape}")
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Log
                    step += 1
                    total_loss += loss.item() * accumulation_steps
                    
                    if step % 10 == 0:
                        avg_loss = total_loss / step
                        logger.info(f"Step {step}/{steps}, Loss: {avg_loss:.4f}")
                    
                    # Clear cache
                    if step % 10 == 0:
                        clear_gpu_memory()
                        
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                logger.error(f"Batch keys: {list(batch.keys())}")
                # Continue with next batch instead of failing
                continue
            
            # Break if we've done enough steps
            if step >= steps:
                break
        
        # Final logging
        avg_loss = total_loss / step if step > 0 else 0
        logger.info(f"Direct fine-tuning completed. Final loss: {avg_loss:.4f}")