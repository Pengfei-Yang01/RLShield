import torch
import numpy as np
import os
import json
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dynamic import find_critical_para, load_model
from fixed import find_critical_para as find_critical_para_fixed


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def calculate_score(model_id, data, safety_layers=None, start_index=0):
    """
    Calculate score for each data using gradient cosine similarity
    
    Args:
        model_id: Model path or ID
        data: Data list read from JSON
        safety_layers: Safety layer index list, if specified only search critical parameters in these layers (default None, use all layers)
        start_index: Start processing from which data (default 0, from beginning)
    
    Returns:
        data: Data list with 'score' field added
    """
    # Load model only once
    model, tokenizer = load_model(model_id)
    
    # Get model device (for moving input_ids to correct device)
    model_device = next(model.parameters()).device
    
    # Define safety layer filter function
    def is_in_safety_layers(param_name, safety_layers):
        """
        Check if parameter is in safety layers
        
        Args:
            param_name: Parameter name, e.g., "model.layers.8.self_attn.q_proj.weight"
            safety_layers: Safety layer index list, e.g., [8, 9, 10, 11, 12, 13]
        
        Returns:
            bool: True if parameter is in safety layers, False otherwise
        """
        if safety_layers is None:
            return True  # If no safety layers specified, use all layers
        
        # Try to extract layer index from parameter name
        # e.g., "model.layers.8.self_attn.q_proj.weight" -> 8
        import re
        match = re.search(r'layers\.(\d+)\.', param_name)
        if match:
            layer_idx = int(match.group(1))
            return layer_idx in safety_layers
        return False  # If cannot extract layer index, exclude (conservative strategy)
    
    # Print information if safety layers are specified
    if safety_layers is not None:
        print("\n" + "="*60)
        print(f"Using specified safety layers: {safety_layers}")
        print(f"Number of safety layers: {len(safety_layers)}")
        print("="*60 + "\n")
    
    # Always use dual filter: calculate fixed reference differences
    print("\n" + "="*60)
    print("Dual filter mode enabled: calculating fixed reference differences...")
    print("="*60)
    _, fixed_minus_row_cos, fixed_minus_col_cos = find_critical_para_fixed(model, tokenizer, model_id, safety_layers=safety_layers)
    print("Fixed reference calculation completed\n")
    
    # Determine model type based on model_id and select corresponding prompt template and sep_token
    print(f"Model ID: {model_id}")
    
    if 'phi-3' in model_id.lower() or 'phi3' in model_id.lower():
        # Phi-3 model (including Phi-3-mini-4k-instruct)
        print("Detected Phi-3 model")
        sep_token = '<|placeholder1|>'
        sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
        unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None
        if sep_token_id is None or (unk_token_id is not None and sep_token_id == unk_token_id):
            # If <|placeholder1|> doesn't exist or is invalid, use eos_token as fallback
            sep_token = tokenizer.eos_token if tokenizer.eos_token else '<|endoftext|>'
            sep_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.convert_tokens_to_ids('<|endoftext|>')
            if sep_token_id is None:
                raise ValueError(f"Cannot find valid sep_token for Phi-3 model. model_id={model_id}")
        prompt = (
            f'<|system|>\n{{system_prompt}}<|end|>\n<|user|>\n{{content}}<|end|>\n<|assistant|>\n{{sep_token}}{{summary}}<|end|>'
        )
    elif 'Qwen' in model_id or 'qwen' in model_id:
        # Qwen2.5 model
        print("Detected Qwen2.5 model")
        sep_token = '<|quad_start|>'
        sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
        if sep_token_id == tokenizer.unk_token_id:
            # If <|quad_start|> doesn't exist, use <|im_end|>
            sep_token = '<|im_end|>'
            sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
        prompt = (
            f'<|im_start|>system\n{{system_prompt}}<|im_end|>\n<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n{{sep_token}}{{summary}}<|im_end|>'
        )
    elif 'llama-3.1' in model_id.lower() or 'llama3.1' in model_id.lower() or 'llama-3.2' in model_id.lower() or 'llama3.2' in model_id.lower():
        # Llama-3.1/3.2 model (using same template format)
        model_version = 'Llama-3.2' if ('llama-3.2' in model_id.lower() or 'llama3.2' in model_id.lower()) else 'Llama-3.1'
        print(f"Detected {model_version} model")
        sep_token = '<|python_tag|>'
        sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
        # Check if sep_token_id is valid (need to consider unk_token_id may be None)
        unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None
        if sep_token_id is None or (unk_token_id is not None and sep_token_id == unk_token_id):
            # If <|python_tag|> doesn't exist, raise error and exit
            raise ValueError(f"<|python_tag|> token doesn't exist or is invalid. sep_token_id={sep_token_id}, unk_token_id={unk_token_id}. Please check if model supports this token.")
        prompt = (
            f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{system_prompt}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{sep_token}}{{summary}}<|eot_id|>'
        )
    else:
        # Llama-2 model (default)
        print("Detected Llama-2 model")
        if tokenizer.unk_token_id is not None:
            sep_token, sep_token_id = tokenizer.unk_token, tokenizer.unk_token_id
        else:
            # If unk_token is also None, use eos_token
            sep_token, sep_token_id = tokenizer.eos_token, tokenizer.eos_token_id
            if sep_token_id is None:
                raise ValueError(f"Cannot find valid sep_token, tokenizer.unk_token_id={tokenizer.unk_token_id}, eos_token_id={tokenizer.eos_token_id}")
        prompt = (
            f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]' + 
            f'{{sep_token}} {{summary}} {{eos_token}}'
        )
    
    print(f"Using sep_token: {sep_token} (ID: {sep_token_id})\n")
    
    # Define inner function apply_prompt_template to format sample data into complete prompt
    def apply_prompt_template(sample):
        if 'phi-3' in model_id.lower() or 'phi3' in model_id.lower():
            # Phi-3 model: doesn't include eos_token parameter, uses <|end|> as end marker
            txt = prompt.format(
                system_prompt='You are a helpful assistant. Help me with the following query: ',
                content=sample['source'],
                summary=sample['target'],
                sep_token=sep_token,
            )
        elif 'llama-3.1' in model_id.lower() or 'llama3.1' in model_id.lower() or 'llama-3.2' in model_id.lower() or 'llama3.2' in model_id.lower():
            # Llama-3.1/3.2 model: doesn't include eos_token parameter, uses <|eot_id|> as end marker
            txt = prompt.format(
                system_prompt='You are a helpful assistant. Help me with the following query: ',
                content=sample['source'],
                summary=sample['target'],
                sep_token=sep_token,
            )
        elif 'Qwen' in model_id or 'qwen' in model_id:
            # Qwen2.5 model: doesn't include eos_token parameter
            txt = prompt.format(
                system_prompt='You are a helpful assistant. Help me with the following query: ',
                content=sample['source'],
                summary=sample['target'],
                sep_token=sep_token,
            )
        else:
            # Llama-2 model: includes eos_token parameter
            txt = prompt.format(
                system_prompt='You are a helpful assistant. Help me with the following query: ',
                content=sample['source'],
                summary=sample['target'],
                eos_token=tokenizer.eos_token,
                sep_token=sep_token,
            )
        # Return dictionary containing formatted text
        return {
            'text': txt,
        }     
    
    # If start index is specified, only process data from start_index
    if start_index > 0:
        print(f"Starting from data {start_index + 1} (index starts from 0)")
        data_to_process = data[start_index:]
    else:
        data_to_process = data
    
    # Record failed samples
    failed_samples = []
    
    # Iterate through each record in JSON data, use tqdm to show progress
    for local_idx, item in enumerate(tqdm(data_to_process, desc="Calculating score")):
        # Calculate global index
        idx = start_index + local_idx
        
        # Use try-except to catch errors when processing individual samples, skip problematic samples
        try:
            # Get safe_reference and unsafe_reference for this sample
            safe_ref = item.get('safe_reference', [])
            unsafe_ref = item.get('unsafe_reference', [])
            
            # Ensure list type to maintain order (if set or other type, convert to list first)
            if isinstance(safe_ref, (set, tuple)):
                safe_ref = list(safe_ref)
            if isinstance(unsafe_ref, (set, tuple)):
                unsafe_ref = list(unsafe_ref)
        
            # Use actual safe_reference, ensure string list (use all references)
            safe_set = [str(ref) for ref in safe_ref if ref]
            
            # Use actual unsafe_reference, ensure string list (use all references)
            unsafe_set = [str(ref) for ref in unsafe_ref if ref]
            
            # Calculate critical parameters for this sample (dynamically use corresponding safe/unsafe references)
            # Pass pre-loaded model and tokenizer to avoid repeated loading
            gradient_norms_compare, minus_row_cos, minus_col_cos = find_critical_para(
                model, tokenizer, model_id, unsafe_set=unsafe_set, safe_set=safe_set, safety_layers=safety_layers
            )

            # Create SGD optimizer for computing gradients
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            # Initialize basic sample dictionary for building prompt
            basic_sample = {}
            # Set current sample's prompt as source field
            basic_sample["source"] = item.get('prompt', '')
            # Set target response as "Sure" (a simple compliance response)
            basic_sample["target"] = "Sure"
            # Apply prompt template to get formatted text
            d = apply_prompt_template(basic_sample)
            # Use tokenizer to convert text to token ID list
            input_ids = tokenizer(d['text']).input_ids
            # Safety check: ensure sep_token_id is not None and in input_ids
            if sep_token_id is None:
                raise ValueError(f"sep_token_id is None, cannot continue. model_id={model_id}, sep_token={sep_token}")
            if sep_token_id not in input_ids:
                raise ValueError(f"sep_token_id ({sep_token_id}, token='{sep_token}') not in input_ids. input_ids length={len(input_ids)}, first 10 tokens={input_ids[:10]}, first 100 chars of text: {d['text'][:100]}")
            # Find separator token position in token ID list
            sep = input_ids.index(sep_token_id)
            
            # Remove separator token, keep tokens before and after (skip separator position)
            input_ids = input_ids[:sep] + input_ids[sep+1:]
            # Convert token ID list to 2D tensor, batch size 1, move to model device
            input_ids = torch.tensor(np.array([input_ids]), device=model_device)
            # Clone input_ids as target_ids for computing task loss
            target_ids = input_ids.clone()
            # Mark input part's target IDs as -100 (indicates not computing gradient and loss)
            target_ids[:, :sep] = -100
            # Clear gradient buffer in optimizer
            optimizer.zero_grad()
            # Forward pass, compute model output and loss
            outputs = model(input_ids, labels=target_ids)
            # Get negative log likelihood loss
            neg_log_likelihood = outputs.loss
            # Backward pass, compute gradients for all parameters
            neg_log_likelihood.backward()
            # Clean intermediate variables to free memory
            sync_cuda()
            del outputs, neg_log_likelihood, input_ids, target_ids
            
            # Initialize cosine list
            cos = []
            # Iterate through all parameters and parameter names of model
            for name, param in model.named_parameters():
                # Check if parameter has gradient, name contains "mlp" or "self" (safety-critical parameters), and is in safety layers
                if param.grad is not None and ("mlp" in name or "self" in name) and is_in_safety_layers(name, safety_layers):
                    # Skip if gradient is 1D
                    if param.grad.dim() == 1:
                        continue
                    # Check if parameter exists in all required dictionaries
                    if name not in gradient_norms_compare or name not in minus_row_cos or name not in minus_col_cos:
                        # Skip if parameter not found in reference dictionaries (should not happen, but safe guard)
                        continue
                    # Move gradient to same device as reference gradient (GPU or CPU)
                    # Use detach().clone() to copy gradient, avoid direct reference causing inability to release
                    grad_norm = param.grad.detach().clone().to(gradient_norms_compare[name].device)
                    grad_ref = gradient_norms_compare[name].to(grad_norm.device)
                    
                    # Dynamic reference differences (align devices)
                    ref_row_dynamic = minus_row_cos[name].to(grad_norm.device)
                    ref_col_dynamic = minus_col_cos[name].to(grad_norm.device)
                    
                    # Always use dual filter: use both fixed and dynamic reference differences
                    # Check if parameter name exists in fixed reference
                    if name in fixed_minus_row_cos and name in fixed_minus_col_cos:
                        ref_row_fixed = fixed_minus_row_cos[name].to(grad_norm.device)
                        ref_col_fixed = fixed_minus_col_cos[name].to(grad_norm.device)
                        row_mask = (ref_row_fixed > 1) & (ref_row_dynamic > 0.8)
                        col_mask = (ref_col_fixed > 1) & (ref_col_dynamic > 0.8)
                        if row_mask.any():
                            sync_cuda()
                            row_cos = torch.nan_to_num(
                                F.cosine_similarity(
                                    grad_norm[row_mask],
                                    grad_ref[row_mask],
                                    dim=1,
                                )
                            )
                            sync_cuda()
                            cos.extend(row_cos.cpu().tolist())
                            del row_cos
                        if col_mask.any():
                            sync_cuda()
                            col_cos = torch.nan_to_num(
                                F.cosine_similarity(
                                    grad_norm[:, col_mask],
                                    grad_ref[:, col_mask],
                                    dim=0,
                                )
                            )
                            sync_cuda()
                            cos.extend(col_cos.cpu().tolist())
                            del col_cos
                        del row_mask, col_mask, ref_row_fixed, ref_col_fixed
                    # If parameter not in fixed reference, skip (don't add any cosine similarity)
                    
                    # Immediately delete gradient tensors no longer needed
                    sync_cuda()
                    del grad_norm, grad_ref, ref_row_dynamic, ref_col_dynamic

            # Average cosine similarity values for current sample to get single score
            score = sum(cos) / len(cos) if len(cos) > 0 else 0.0
            
            # Add score to data item
            item['score'] = float(score)
            
            # Clean gradient data no longer needed
            sync_cuda()
            del gradient_norms_compare, minus_row_cos, minus_col_cos
            optimizer.zero_grad()  # Clear gradients
            del optimizer
            del cos
        
        except Exception as e:
            # Catch errors when processing individual samples
            error_info = {
                'index': idx,
                'prompt': item.get('prompt', '')[:100] if item.get('prompt') else 'N/A',  # Only record first 100 chars
                'error': str(e),
                'error_type': type(e).__name__
            }
            failed_samples.append(error_info)
            
            # Print error information
            print(f"\nError processing sample {idx + 1}:", flush=True)
            print(f"   Index: {idx}", flush=True)
            print(f"   Prompt: {error_info['prompt']}", flush=True)
            print(f"   Error type: {error_info['error_type']}", flush=True)
            print(f"   Error message: {error_info['error']}", flush=True)
            
            # Set a default score to avoid subsequent processing errors
            item['score'] = None
            item['error'] = str(e)
            
            # Continue processing next sample
            continue
    
    # Clean model and tokenizer
    sync_cuda()
    del model
    del tokenizer
    
    # Print summary of failed samples
    if failed_samples:
        print(f"\n" + "="*60)
        print(f"Summary of failed samples")
        print("="*60)
        print(f"Number of failed samples: {len(failed_samples)}")
        print(f"Failed sample indices: {[s['index'] for s in failed_samples]}")
        print("\nDetailed error information:")
        for sample in failed_samples:
            print(f"  Index {sample['index']}: {sample['error_type']} - {sample['error']}")
            print(f"    Prompt: {sample['prompt']}")
        print("="*60 + "\n")
    else:
        print(f"\nAll samples processed successfully!\n")
    
    # Return data with 'score' field added
    return data


def process_file(input_path, model_id, safety_layers=None, start_index=0):
    """
    Process single JSON file, calculate score
    
    Args:
        input_path: Input JSON file path
        model_id: Model path
        safety_layers: Safety layer index list
        start_index: Start processing from which data (default 0, from beginning)
    """
    try:
        # Read JSON data
        print(f"\nReading JSON file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Number of samples: {len(data)}")
        
        # Check if start_index is valid
        if start_index < 0:
            print(f"Warning: start_index ({start_index}) is invalid, will start from 0")
            start_index = 0
        if start_index >= len(data):
            print(f"Error: start_index ({start_index}) exceeds data range (0-{len(data)-1})")
            return
        
        # Calculate score
        data_with_scores = calculate_score(
            model_id, data,
            safety_layers=safety_layers,
            start_index=start_index
        )
        
        processed_count = len(data_with_scores) - start_index
        print(f"Processed {processed_count} samples (starting from index {start_index})")
        
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        import traceback
        traceback.print_exc()


# Main program entry point
if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Calculate score for each data')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSON file path (required)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Model path (required)')
    parser.add_argument('--safety_layer', type=str, default=None,
                        help='Safety layer JSON file path. If specified, will only search critical parameters in these layers')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start processing from which data (default 0, from beginning. Index starts from 0)')
    args = parser.parse_args()
    
    # safety_layer - Read safety layer JSON file
    safety_layers = None
    if args.safety_layer is not None:
        try:
            with open(args.safety_layer, 'r', encoding='utf-8') as f:
                safety_data = json.load(f)
                safety_layers = safety_data.get('safety_layers', None)
                if safety_layers:
                    print(f"\nRead safety layers from {args.safety_layer}: {safety_layers}")
                    print(f"Safety layer range: {safety_data.get('safety_layer_range', 'unknown')}")
                else:
                    print(f"\nWarning: 'safety_layers' field not found in {args.safety_layer}, will use all layers")
        except FileNotFoundError:
            print(f"\nError: File {args.safety_layer} not found, will use all layers")
        except json.JSONDecodeError:
            print(f"\nError: {args.safety_layer} is not a valid JSON file, will use all layers")
        except Exception as e:
            print(f"\nError: Error reading {args.safety_layer}: {e}, will use all layers")
    
    # Get input file path
    input_file = args.input_file
    model_path = args.model_path
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        exit(1)
    
    # Print configuration information
    print("\n" + "="*60)
    print("Calculate score")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Model path: {model_path}")
    print(f"Dual filter: Enabled (always)")
    print(f"Vector normalization: Disabled (always)")
    print(f"Safety layer filter: {'Enabled - ' + str(safety_layers) if safety_layers else 'Disabled (use all layers)'}")
    print(f"Start index: {args.start_index} (starting from data {args.start_index + 1})")
    print("="*60 + "\n")
    
    # Process single input file
    process_file(
        input_file,
        model_path,
        safety_layers=safety_layers,
        start_index=args.start_index
    )
    
    print("\nFile processing completed!")