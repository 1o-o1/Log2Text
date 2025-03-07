# DeepSeek GRPO for Authentication Analysis

This project implements a memory-optimized version of DeepSeek's GRPO (Generative Reinforcement from Preference Optimization) for fine-tuning LLMs to analyze authentication logs. The implementation is specifically designed to work on machines with limited GPU memory (8GB) and moderate RAM (16GB).

## Training Pipeline Overview

The training pipeline follows the GRPO methodology with memory optimizations:

```mermaid
flowchart TD
    start([Start]) --> config[Load Configuration]
    config --> tokenizer[Initialize Tokenizer]
    tokenizer --> data_proc[Initialize Data Processor]
    
    subgraph SFT["Supervised Fine-Tuning (SFT)"]
        sft_data[Prepare SFT Dataset with auth logs\nand expert annotations] --> sft_model[Load Base Model]
        sft_model --> add_lora1[Add LoRA Adapter]
        add_lora1 --> sft_train[Train with Custom Loop\n- Loss: Causal LM Loss\n- Optimizer: Adafactor\n- LR: config['learning_rate']]
        sft_train --> sft_save[Save SFT Model]
        
        sft_train -->|"Epoch loss tracking"| sft_log["Log: avg_epoch_loss = epoch_loss / steps\n(SFT Trainer)"]
    end
    
    subgraph RM["Reward Model Training"]
        pref_data[Prepare Preference Dataset\nwith chosen/rejected pairs] --> mem_check{Check Memory\nAvailability}
        mem_check -->|"≥ 7GB"| rm_model[Load Base Classification Model]
        mem_check -->|"< 7GB"| alt_rm[Create Alternative Reward Model\n(metadata-based heuristic)]
        
        rm_model --> add_lora2[Add LoRA Adapter\nfor Sequence Classification]
        add_lora2 --> rm_train[Train with Bradley-Terry Loss\n- Loss: -log(sigmoid(chosen - rejected))\n- Optimizer: AdamW\n- LR: config['learning_rate']]
        rm_train --> rm_save[Save Reward Model]
        
        alt_rm --> rm_save
        
        rm_train -->|"Epoch loss tracking"| rm_log["Log: avg_loss = epoch_loss / num_batches\n(Reward Model Trainer)"]
    end
    
    subgraph RL["Reinforcement Learning (Simplified PPO)"]
        rl_data[Load High-Quality\nExamples] --> load_sft[Load SFT Model]
        load_sft --> add_lora3[Load/Add LoRA Adapter]
        add_lora3 --> direct_ft[Simplified PPO\n- Loss: Causal LM Loss\n- Optimizer: Adafactor\n- LR: config['learning_rate'] / 2]
        direct_ft --> rl_save[Save RL Model]
        
        direct_ft -->|"Step loss tracking"| rl_log["Log: avg_loss = total_loss / step\n(RL Trainer)"]
    end
    
    data_proc --> SFT
    SFT --> RM
    RM --> RL
    
    RL --> model_ready([Trained Model Ready for Inference])
    
    model_ready --> inference[Run Inference\nGenerate Authentication Analysis]
```

### Training Steps

1. **Supervised Fine-Tuning (SFT)**
   - Uses authentication logs paired with expert JSON annotations
   - Implements custom training loop with memory optimizations
   - Tracks average epoch loss as `epoch_loss / steps`
   - Uses Adafactor optimizer with configurable learning rate

2. **Reward Model Training**
   - Creates preference pairs (chosen/rejected examples)
   - Uses Bradley-Terry loss: `-log(sigmoid(chosen_rewards - rejected_rewards))`
   - Performs memory check - uses different approaches based on available GPU memory
   - Falls back to a heuristic-based model if memory is insufficient

3. **Reinforcement Learning (Simplified PPO)**
   - Implements a memory-efficient version of Proximal Policy Optimization
   - Uses the SFT model as starting point
   - Trains on high-quality examples with reduced learning rate
   - Limits training to a small number of steps (50) to prevent overfitting

4. **Inference**
   - Uses the final trained model to generate structured authentication analysis
   - Implements robust JSON extraction for reliable outputs

## Project Structure

The codebase is organized into modular components:

```
├── config.py           # Configuration settings and utilities
├── data_processor.py   # Data loading and preprocessing
├── model_utils.py      # Model initialization and utilities
├── sft_trainer.py      # Supervised Fine-Tuning module
├── reward_trainer.py   # Reward model training module
├── rl_trainer.py       # Reinforcement Learning module
├── inference.py        # Inference functionality
├── main.py             # Main script to run the pipeline
└── schema.json         # JSON schema for authentication analysis
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU with at least 8GB VRAM (will fallback to CPU if needed)
- 16GB+ system RAM

### Python Dependencies

```
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
trl>=0.4.7
accelerate>=0.20.0
bitsandbytes>=0.39.0
datasets>=2.12.0
pandas>=1.5.0
numpy>=1.24.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/deepseek-auth-analysis.git
cd deepseek-auth-analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Format

The system expects data in a specific directory structure:

```
data/
├── auth_data/
│   └── benign/
│       ├── auth_data_1_165.txt
│       ├── auth_data_2_166.txt
│       └── ...
└── Auth_bin/
    ├── auth_text_1_165.json
    ├── auth_text_2_166.json
    └── ...
```

Each `auth_data_X_Y.txt` file contains raw authentication logs, while the corresponding `auth_text_X_Y.json` contains the expert-annotated security analysis in JSON format.

## Usage

### Training

To run the complete training pipeline:

```bash
python main.py train --data_dir /path/to/data --output_dir ./models --samples 100
```

To train only a specific stage:

```bash
# Train only the SFT model
python main.py train --data_dir /path/to/data --stage sft

# Train only the reward model
python main.py train --data_dir /path/to/data --stage reward

# Train only the RL model
python main.py train --data_dir /path/to/data --stage rl
```

If you have very limited GPU memory, you can force CPU training (very slow but will complete):

```bash
python main.py train --data_dir /path/to/data --cpu
```

### Inference

To analyze authentication data using a trained model:

```bash
python main.py infer --model_dir ./models --input_file auth_data.txt --output_file analysis.json
```

To use the full schema in the prompt (may improve results at the cost of more tokens):

```bash
python main.py infer --model_dir ./models --input_file auth_data.txt --full_schema
```

## Memory Optimization Techniques

This implementation uses several techniques to run on hardware with limited memory:

1. **4-bit Quantization**: Reduces model memory footprint by 8x
2. **Parameter-Efficient Fine-Tuning**: Uses LoRA with small rank (r=4)
3. **Gradient Accumulation**: Enables training with very small batch sizes
4. **Reduced Sequence Length**: Uses 1024 tokens instead of 2048
5. **CPU Offloading**: Places some model layers on CPU
6. **Simplified PPO**: Uses memory-efficient policy optimization instead of full PPO
7. **Custom Training Loops**: Avoids memory issues with Hugging Face Trainer
8. **Memory Clearing**: Aggressively cleans up GPU memory
9. **Alternative Fallbacks**: Uses simplified approaches when full training is infeasible
10. **Reduced Dataset Size**: Limits training to small subsets of data

## Loss Calculation Details

1. **SFT Loss**: Standard causal language modeling loss
   ```python
   # From sft_trainer.py
   outputs = model(**batch)
   loss = outputs.loss / accumulation_steps
   epoch_loss += loss.item() * accumulation_steps
   avg_epoch_loss = epoch_loss / steps
   ```

2. **Reward Model Loss**: Bradley-Terry loss for preference learning
   ```python
   # From reward_trainer.py
   chosen_rewards = chosen_outputs.logits.squeeze()
   rejected_rewards = rejected_outputs.logits.squeeze()
   loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards))
   loss = loss.mean() / grad_acc_steps
   epoch_loss += loss.item() * grad_acc_steps
   avg_loss = epoch_loss / num_batches
   ```

3. **Simplified PPO Loss**: Policy optimization with simplified objective
   ```python
   # From rl_trainer.py
   outputs = model(**batch)
   loss = outputs.loss / accumulation_steps
   total_loss += loss.item() * accumulation_steps
   avg_loss = total_loss / step
   ```

## Extending the Project

- **Different Models**: To use a different base model, modify the `model_name` in `config.py`
- **Custom Schemas**: Edit `schema.json` to change the output format
- **Additional Tasks**: The GRPO framework can be adapted to other analysis tasks

## Troubleshooting

### Common Issues

- **Out of Memory (OOM) Errors**: Try reducing `max_train_samples` or use `--cpu` flag
- **CUDA Error: device-side assert triggered**: Check your CUDA and PyTorch versions for compatibility
- **Poor JSON Generation**: Consider increasing training epochs or using `--full_schema` during inference
- **Slow Training**: CPU training is extremely slow; consider using Google Colab or other cloud GPUs

## License

This project is released under the MIT License.git init