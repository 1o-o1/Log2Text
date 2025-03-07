import os
import json
import torch
import logging
from config import JSON_SCHEMA_SHORT, load_full_schema
from model_utils import load_tokenizer, load_adapter_model, clear_gpu_memory

logger = logging.getLogger(__name__)

class Inference:
    """
    Class for running inference with trained models to generate authentication analysis.
    """
    
    def __init__(self, config):
        """
        Initialize inference module.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.tokenizer = load_tokenizer(config["model_name"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference will use device: {self.device}")
    
    def generate_analysis(self, auth_data, model_path=None, use_full_schema=True):
        """
        Generate security analysis for authentication data.
        
        Args:
            auth_data (str): Authentication data to analyze
            model_path (str, optional): Path to model. Defaults to RL model.
            use_full_schema (bool): Whether to use full schema in prompt
            
        Returns:
            dict: JSON security analysis
        """
        try:
            # Use specified model path or default to RL model, falling back to SFT
            model_path = model_path or self.config["rl_model_path"]
            
            # Check if the model exists - looking for safetensors instead of bin
            model_exists = (
                os.path.exists(os.path.join(model_path, "adapter_model.safetensors")) or 
                os.path.exists(os.path.join(model_path, "adapter_config.json"))
            )
            
            if not model_exists:
                logger.info(f"Model not found at {model_path}, falling back to SFT model")
                model_path = self.config["sft_model_path"]
                
                # Check if SFT model exists
                sft_exists = (
                    os.path.exists(os.path.join(model_path, "adapter_model.safetensors")) or 
                    os.path.exists(os.path.join(model_path, "adapter_config.json"))
                )
                
                if not sft_exists:
                    logger.info("SFT model not found, using base model")
                    model_path = self.config["model_name"]
            
            # Load schema
            schema_json = load_full_schema() if use_full_schema else JSON_SCHEMA_SHORT
            
            # Parse schema to extract structure but make it more concise
            try:
                schema_obj = json.loads(schema_json)
                # Create a simplified schema representation
                simplified_schema = {
                    "log_type": "Authentication",
                    "field_descriptions": "Details about source/destination computers, auth types, etc.",
                    "observations": {
                        "source_actor": "Who initiated auth",
                        "targets": "Target systems",
                        "temporal_patterns": "Time-based patterns",
                        "behavioral_patterns": "Behavior analysis"
                    },
                    "potential_indicators": {
                        "suspicious_auth_types": "List suspicious auth types",
                        "account_patterns": "Suspicious account usage",
                        "logon_logoff_sequences": "Unusual logon sequences",
                        "anomalies": "Other anomalies"
                    },
                    "next_steps_for_validation": "How to validate findings",
                    "conclusion": "Overall assessment and recommendations",
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
                }
                schema_ref = json.dumps(simplified_schema, indent=2)
            except:
                # If parsing fails, use a hardcoded simplified schema
                schema_ref = """{
  "log_type": "Authentication",
  "field_descriptions": { ... },
  "observations": {
    "source_actor": "Analysis of who initiated auth",
    "targets": { "frequent_targets": [], "sporadic_targets": [] },
    "temporal_patterns": { "clusters": "", "bursts": "", "off_hours_activity": "" },
    "behavioral_patterns": { "repetitive_actions": "", "lateral_movement": "", "privilege_escalation": "" }
  },
  "potential_indicators": {
    "suspicious_auth_types": { "description": "", "affected_entities": [] },
    "account_patterns": { "description": "", "affected_accounts": [] },
    "logon_logoff_sequences": { "description": "", "affected_entities": [] },
    "anomalies": { "description": "", "deviation_details": "" }
  },
  "next_steps_for_validation": { ... },
  "conclusion": {
    "summary": "",
    "recommended_actions": ""
  },
  "high_risk_indicators": {
    "anonymous_logon_detected": false,
    "unknown_auth_type": false,
    "ntlm_in_kerberos_env": false,
    "machine_account_anomalies": false,
    "multiple_accounts_single_source": false,
    "lateral_movement_indicators": false,
    "excessive_ticket_requests": false,
    "incomplete_session_pairs": false
  }
}"""
            
            # Create cybersecurity analysis prompt with simpler output format
            prompt = f"""You are a cybersecurity analyst tasked with analyzing authentication log events to identify potentially suspicious patterns indicative of a security compromise or lateral movement.

## Authentication Data to Analyze:
{auth_data}

## Your Analysis Task:
Analyze this authentication data to identify suspicious patterns, unusual behavior, and potential security threats.

Focus on:
1. Source/destination of authentication attempts
2. Authentication types and protocols used
3. Temporal patterns (time-based anomalies)
4. Behavioral patterns (lateral movement, credential harvesting)

## High-Risk Indicators to Check:
- ANONYMOUS LOGON events
- Unknown authentication types
- NTLM usage in Kerberos environments
- Machine account anomalies
- Multiple accounts from single source
- Lateral movement patterns
- Excessive ticket requests
- Logons without corresponding logoffs

## Response Format:
First, provide your analysis under <THINKING> tags, explaining your reasoning step by step.

Then, provide a structured JSON response following this schema:
{schema_ref}

Each field should contain your specific observations and findings from the authentication data. Be concise but thorough in your analysis.
"""
            
            # Load model and generate
            result = self._generate_with_model(model_path, prompt)
            return result
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {"error": f"Generation failed: {str(e)}"}
    
    def _generate_with_model(self, model_path, prompt, max_tokens=4000):
        """
        Generate text with model.
        
        Args:
            model_path (str): Model path
            prompt (str): Prompt text
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: Parsed JSON result and thinking process
        """
        logger.info(f"Generating analysis with model from {model_path}")
        
        # Tokenize prompt - increased max length
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=3072)
        
        # Check for adapter existence - look for safetensors first, then config
        is_adapter = (
            os.path.exists(os.path.join(model_path, "adapter_model.safetensors")) or 
            os.path.exists(os.path.join(model_path, "adapter_config.json"))
        )
        
        if is_adapter:
            # Get base model name from config
            base_model = self.config["model_name"]
            logger.info(f"Loading adapter model from {model_path} using base model {base_model}")
            model = load_adapter_model(base_model, model_path, device="auto")
        else:
            # Direct model loading
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            logger.info(f"Loading base model directly from {model_path}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Move inputs to model's device
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate with parameters tuned for structured output
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.2,  # Lower temperature for more structured output
                top_p=0.9,
                repetition_penalty=1.2,  # Add repetition penalty to avoid loops
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Get generated output (excluding prompt)
        generated_ids = outputs[0][len(input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up
        del model, inputs, input_ids, attention_mask, outputs, generated_ids
        clear_gpu_memory()
        
        print(response)
        # Extract thinking process
        thinking = ""
        thinking_start = response.find("</think>")
        thinking_end = response.find("</think>")
        
        if thinking_start != -1 and thinking_end != -1 and thinking_end > thinking_start:
            thinking = response[thinking_start+10:thinking_end].strip()
            # Remove the thinking part from the response for JSON parsing
            response = response[:thinking_start] + response[thinking_end+11:]
        
        # Extract JSON part
        result = self._extract_json(response, thinking)
        return result
    
    def _extract_json(self, response, thinking=""):
        """Helper method to extract and parse JSON from the response"""
        try:
            # First attempt: find standard JSON
            json_start = response.find("{")
            json_end = response.rfind("}")
            
            if json_start == -1 or json_end == -1 or json_end <= json_start:
                logger.error("Failed to find valid JSON content in model response")
                return {
                    "error": "No valid JSON content found in response", 
                    "raw_response": response,
                    "thinking": thinking
                }
            
            json_text = response[json_start:json_end+1]
            
            # Attempt to parse the JSON
            try:
                analysis = json.loads(json_text)
                if thinking:
                    analysis["thinking_process"] = thinking
                return analysis
            except json.JSONDecodeError:
                # Clean up common JSON issues
                logger.info("Initial JSON parsing failed, attempting cleanup...")
                
                # Remove embedded newlines in string values
                json_text = re.sub(r'"\s*\n\s*(.+?)\s*\n\s*"', r'"\1"', json_text)
                
                # Fix trailing commas
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)
                
                # Replace invalid control characters
                json_text = re.sub(r'[\x00-\x1F]', '', json_text)
                
                # Handle nested quotes
                json_text = json_text.replace('\\"', '"')
                
                try:
                    analysis = json.loads(json_text)
                    if thinking:
                        analysis["thinking_process"] = thinking
                    return analysis
                except json.JSONDecodeError as e:
                    # Last resort - construct a basic JSON with the error and raw response
                    logger.error(f"JSON parsing failed after cleanup: {str(e)}")
                    return {
                        "error": "Failed to parse JSON", 
                        "json_decode_error": str(e),
                        "raw_response": response,
                        "partial_json": json_text,
                        "thinking": thinking
                    }
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return {
                "error": str(e), 
                "raw_response": response,
                "thinking": thinking
            }