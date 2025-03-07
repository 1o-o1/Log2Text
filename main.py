import os
import json
import logging
import argparse
import torch

from config import create_config
from data_processor import DataProcessor
from model_utils import load_tokenizer, clear_gpu_memory
from sft_trainer import SFTTrainer
from reward_trainer import RewardModelTrainer
from rl_trainer import RLTrainer
from inference import Inference

logger = logging.getLogger(__name__)

def check_system_resources():
    """Check system resources and provide warnings if necessary"""
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be very slow on CPU.")
        return False
    
    # Check GPU memory
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU has {gpu_memory_gb:.1f} GB of memory")
        
        if gpu_memory_gb < 8:
            logger.warning(f"Available GPU memory ({gpu_memory_gb:.1f} GB) is very limited. "
                          "Training may fail or be extremely slow.")
            return False
    except Exception as e:
        logger.warning(f"Could not check GPU memory: {str(e)}")
        return False
    
    # Check CPU memory (approximate)
    try:
        import psutil
        cpu_memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"System has {cpu_memory_gb:.1f} GB of RAM")
        
        if cpu_memory_gb < 16:
            logger.warning(f"Available system memory ({cpu_memory_gb:.1f} GB) is limited. "
                          "You may experience out-of-memory issues.")
            return False
    except:
        logger.warning("Could not check system memory")
    
    return True

def run_training_pipeline(config, only_stage=None):
    """
    Run the training pipeline with the specified configuration.
    
    Args:
        config (dict): Configuration dictionary
        only_stage (str, optional): Run only a specific stage ('sft', 'reward', 'rl')
        
    Returns:
        bool: Success status
    """
    try:
        # Initialize tokenizer
        tokenizer = load_tokenizer(config["model_name"])
        
        # Initialize data processor
        data_processor = DataProcessor(config, tokenizer)
        
        # Create trainers
        sft_trainer = SFTTrainer(config, tokenizer, data_processor)
        reward_trainer = RewardModelTrainer(config, tokenizer, data_processor)
        rl_trainer = RLTrainer(config, tokenizer, data_processor)
        
        # Run training pipeline based on stage
        if only_stage == 'sft' or only_stage is None:
            logger.info("=== Starting Supervised Fine-Tuning ===")
            sft_model_path = sft_trainer.train()
            logger.info(f"SFT model saved to {sft_model_path}")
            clear_gpu_memory()
        
        if only_stage == 'reward' or only_stage is None:
            logger.info("=== Starting Reward Model Training ===")
            reward_model_path = reward_trainer.train()
            logger.info(f"Reward model saved to {reward_model_path}")
            clear_gpu_memory()
        
        if only_stage == 'rl' or only_stage is None:
            logger.info("=== Starting RL Fine-Tuning ===")
            rl_model_path = rl_trainer.train()
            logger.info(f"RL model saved to {rl_model_path}")
            clear_gpu_memory()
        
        logger.info("Training pipeline completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_inference(config, auth_data, use_full_schema=False):
    """
    Run inference with the trained model.
    
    Args:
        config (dict): Configuration dictionary
        auth_data (str): Authentication data to analyze
        use_full_schema (bool): Whether to use full schema
        
    Returns:
        dict: Generated analysis
    """
    inference = Inference(config)
    return inference.generate_analysis(auth_data, use_full_schema=use_full_schema)

def main():
    """Main function to parse arguments and run the appropriate action"""
    parser = argparse.ArgumentParser(description="DeepSeek GRPO for Authentication Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data_dir", required=True, help="Directory with training data")
    train_parser.add_argument("--output_dir", default="./models", help="Directory to save models")
    train_parser.add_argument("--stage", choices=["sft", "reward", "rl"], help="Train only specific stage")
    train_parser.add_argument("--samples", type=int, default=100, help="Max number of training samples")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU training (slow)")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model_dir", required=True, help="Directory with trained models")
    infer_parser.add_argument("--input_file", required=True, help="Input file with auth data")
    infer_parser.add_argument("--output_file", help="Output file for analysis")
    infer_parser.add_argument("--full_schema", action="store_true", help="Use full schema in prompt")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Check system resources if not forcing CPU
        if not args.cpu:
            if not check_system_resources():
                logger.warning("System resources may be insufficient. Use --cpu to force CPU training.")
        
        # Create configuration
        config = create_config(args.data_dir, args.output_dir)
        
        # Override max samples
        if args.samples:
            config["max_train_samples"] = args.samples
        
        # Force CPU if requested
        if args.cpu:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("Forcing CPU training (will be slow)")
        
        # Run training pipeline
        run_training_pipeline(config, only_stage=args.stage)
    
    elif args.command == "infer":
        import os
        # Load config or create a new one
        config_path = os.path.join(args.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = create_config("", args.model_dir)
        
        # Read input file
        with open(args.input_file, "r") as f:
            auth_data = f.read()
        
        # Run inference
        result = run_inference(config, auth_data, use_full_schema=args.full_schema)
        
        # Output result
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Analysis saved to {args.output_file}")
        else:
            print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
