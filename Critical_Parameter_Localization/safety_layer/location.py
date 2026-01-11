"""
Compute cosine similarity between sentence pairs and locate safety layers

This script analyzes the similarity between safe and unsafe samples in the model's hidden state space.
By computing cosine similarity of different sentence pairs at each layer's hidden state vectors,
we study how the model processes different types of inputs.


"""

import os
import sys
sys.path.append("..")  # Add parent directory to path for importing utils module

import json
import warnings

import numpy as np
import argparse
import torch

from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from utils.prompter import Prompter

warnings.filterwarnings('ignore')


def load_pairs_from_json(json_path: str) -> list:
    """
    Load sentence pairs from JSON file
    
    Args:
        json_path: JSON file path, should contain 'pairs' field, each pair has 'sentence1' and 'sentence2'
    
    Returns:
        pairs: List of sentence pairs, each element is [sentence1, sentence2]
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'pairs' not in data:
        raise ValueError(f"JSON file format error: missing 'pairs' field: {json_path}")
    
    pairs = []
    for pair_data in data['pairs']:
        if 'sentence1' not in pair_data or 'sentence2' not in pair_data:
            raise ValueError(f"JSON file format error: pair missing 'sentence1' or 'sentence2' field")
        pairs.append([pair_data['sentence1'], pair_data['sentence2']])
    
    print(f"Loaded {len(pairs)} sentence pairs from {json_path}")
    return pairs


def get_output(
    model,
    instruction,
    prompter,
    tokenizer,
    input=None,
    temperature=0.5,
    top_p=0.2,
    top_k=40,
    num_beams=4,
    max_new_tokens=1,
    device='cuda'
):
    prompt = prompter.generate_prompt(instruction, input)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=0
    )

    generation_output = model.generate(
        input_ids=input_ids,
        output_hidden_states=True,
        generation_config=generation_config,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )
    return generation_output


def get_r_lists_cossim_from_pairs(model, prompter, tokenizer, pairs):
    """
    Compute cosine similarity at each layer using predefined sentence pairs
    
    Design:
    - Use saved sentence pair list
    - For each pair, extract the last token's hidden state vector at each layer
    - Compute cosine similarity between two vectors at each layer
    - Return similarity matrix of shape [len(pairs), num_layers]
    
    Args:
        model: Language model
        prompter: Prompt generator
        tokenizer: Tokenizer
        pairs: List of sentence pairs, each element is [sentence1, sentence2]
    
    Returns:
        allcos: List of lists, each inner list contains cosine similarity of a sentence pair at each layer
    """
    allcos = []

    for idx, (instruction1, instruction2) in enumerate(pairs):
        if (idx + 1) % 100 == 0:
            print(f"  Processing progress: {idx + 1}/{len(pairs)}")

        # First sentence: extract hidden state vectors at each layer
        # Note: skip layer 0 (usually embedding layer), take the last token's vector at each layer
        all_vectors = []
        generation_output1 = get_output(
            model=model,
            instruction=instruction1,
            prompter=prompter,
            tokenizer=tokenizer
        )
        hs1 = generation_output1['hidden_states']
        for i in range(len(hs1[0])):
            if i == 0:  
                continue
            # hs1[0][i] is the hidden states of all tokens at layer i
            # [0][-1] takes the vector of the last token of the first sequence
            all_vectors.append(hs1[0][i][0][-1])

        # Second sentence: similarly extract hidden state vectors at each layer
        all_vectors2 = []
        generation_output2 = get_output(
            model=model,
            instruction=instruction2,
            prompter=prompter,
            tokenizer=tokenizer
        )
        hs2 = generation_output2['hidden_states']
        for i in range(len(hs2[0])):
            if i == 0: 
                continue
            all_vectors2.append(hs2[0][i][0][-1])

        # Compute cosine similarity layer by layer
        cso = []
        for k in range(len(all_vectors2)):
            a = all_vectors[k].cpu().detach().numpy()
            b = all_vectors2[k].cpu().detach().numpy()
            # Cosine similarity formula: cos(θ) = (a·b) / (||a|| * ||b||)
            cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cso.append(float(cosine_similarity))

        allcos.append(cso)

    print(f'Computation completed, processed {len(allcos)} sentence pairs')
    return allcos


def locate_safety_layers(allcos_SS, allcos_SU, epsilon=0.0001):
    """
    Args:
        allcos_SS: Cosine similarity list of S-S pairs, shape [r, num_layers]
        allcos_SU: Cosine similarity list of S-U pairs, shape [r, num_layers]
        epsilon: Small perturbation threshold in statistical phase for judging whether cos_gap increases (default 0.0001)
    
    Returns:
        safety_layers: List of identified safety layer indices
        layer_analysis: Dictionary containing detailed analysis data for each layer
    """
    # Convert to numpy array for computation
    cos_SS = np.array(allcos_SS)  # [r, num_layers]
    cos_SU = np.array(allcos_SU)  # [r, num_layers]

    # Compute statistics for each layer
    mean_cos_SS = np.mean(cos_SS, axis=0)  # Average similarity of S-S pairs at each layer
    mean_cos_SU = np.mean(cos_SU, axis=0)  # Average similarity of S-U pairs at each layer
    std_cos_SS = np.std(cos_SS, axis=0)    # Standard deviation of S-S pairs at each layer
    std_cos_SU = np.std(cos_SU, axis=0)    # Standard deviation of S-U pairs at each layer

    # Core metric: cos_gap represents model's ability to distinguish safe/unsafe queries
    # Larger cos_gap indicates more obvious distinction of safe/unsafe queries at this layer
    cos_gap = mean_cos_SS - mean_cos_SU

    num_layers = len(cos_gap)


    safety_layers = []
    started = False 
    
    for i in range(1, num_layers-1):
        if not started:

            if cos_gap[i] - cos_gap[i - 1] > 0.001:
                started = True

                safety_layers.append(i)
        else:
            if cos_gap[i] > cos_gap[i - 1] + epsilon:
                safety_layers.append(i)

    # ========== Build detailed analysis results ==========
    # Save detailed statistical information for each layer for subsequent analysis and visualization
    layer_analysis = {
        'layer_indices': list(range(num_layers)),  # Layer indices
        'mean_cos_SS': mean_cos_SS.tolist(),      # Average similarity of S-S pairs at each layer
        'mean_cos_SU': mean_cos_SU.tolist(),      # Average similarity of S-U pairs at each layer
        'std_cos_SS': std_cos_SS.tolist(),        # Standard deviation of S-S pairs at each layer
        'std_cos_SU': std_cos_SU.tolist(),        # Standard deviation of S-U pairs at each layer
        'cos_gap': cos_gap.tolist(),              # Core metric: cos_gap at each layer
    }

    return safety_layers, layer_analysis


def main(
    model_path: str = '/path/to/model',
    save_dir: str = '/path/to/results',
    pairs_json_dir: str = None
):
    """
    Main function: Safety layer localization analysis pipeline
    
    Workflow:
    1. Data loading: Load sentence pairs from JSON files
    2. Model loading: Load pretrained language model and tokenizer
    3. Similarity computation:
       - Compute cosine similarity of S-S sentence pairs at each layer using saved pairs
       - Compute cosine similarity of S-U sentence pairs at each layer using saved pairs
    4. Safety layer localization
    5. Result saving: Save detailed analysis results and summary
    
    Design notes:
    - Must use saved sentence pairs from JSON files
    - If pairs_json_dir is None, automatically search for JSON files in save_dir
    - Results include detailed statistical information for each layer for subsequent analysis and visualization
    
    Args:
        model_path: Pretrained language model path
        save_dir: Result saving directory
        pairs_json_dir: Directory containing sentence pair JSON files (if None, search in save_dir)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Determine JSON file paths
    if pairs_json_dir is None:
        pairs_json_dir = save_dir
    
    ss_json_path = os.path.join(pairs_json_dir, 'safe-safe_pairs.json')
    su_json_path = os.path.join(pairs_json_dir, 'safe-unsafe_pairs.json')
    
    # Check if JSON files exist
    if not os.path.exists(ss_json_path):
        raise FileNotFoundError(f"S-S sentence pair JSON file not found: {ss_json_path}")
    if not os.path.exists(su_json_path):
        raise FileNotFoundError(f"S-U sentence pair JSON file not found: {su_json_path}")
    
    print(f"Loading sentence pairs from JSON files")
    print(f"  S-S pair file: {ss_json_path}")
    print(f"  S-U pair file: {su_json_path}")
    
    pairs_SS = load_pairs_from_json(ss_json_path)
    pairs_SU = load_pairs_from_json(su_json_path)
    
    print(f"\nS-S pair count: {len(pairs_SS)}")
    print(f"S-U pair count: {len(pairs_SU)}")

    print(f"\nLoading model: {model_path}")
    prompter = Prompter("alpaca")
    device_map = 'auto'

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Compatible with different architectures: some models have model.model.layers, some may have model.layers
    total_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        total_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        total_layers = len(model.layers)

    if total_layers is not None:
        print(f"Model loaded, total layers: {total_layers}")
    else:
        print("Model loaded (failed to automatically parse total layers field, but does not affect subsequent computation)")

    print(f"\nStarting cosine similarity computation...")

    print(f"\n[1/2] Computing cosine similarity of S-S pairs (total {len(pairs_SS)} pairs)...")
    allcos_SS = get_r_lists_cossim_from_pairs(
        model, prompter, tokenizer, pairs_SS
    )

    print(f"\n[2/2] Computing cosine similarity of S-U pairs (total {len(pairs_SU)} pairs)...")
    allcos_SU = get_r_lists_cossim_from_pairs(
        model, prompter, tokenizer, pairs_SU
    )

    print(f"\nCosine similarity computation completed!")

    model_name_lower = model_path.lower()
    if 'llama3.2' in model_name_lower or 'llama-3.2' in model_name_lower:
        epsilon = 0.0003

    else:
        epsilon = 0.0001


    safety_layers, layer_analysis = locate_safety_layers(allcos_SS, allcos_SU, epsilon=epsilon)

    print(f"\nSafety layer localization completed!")
    print(f"Identified safety layers: {safety_layers}")

    if len(safety_layers) > 0:
        print(f"Safety layer range: Layer {min(safety_layers)} to Layer {max(safety_layers)}")
    else:
        print("No safety layers identified (cos_gap did not show layer-by-layer increase according to current rules)")

    safety_layers_result = {
        'model_name': model_path,
        'total_layers': int(total_layers) if total_layers is not None else None,
        'safety_layers': safety_layers,
        'safety_layer_range': None if len(safety_layers) == 0 else {
            'start': int(min(safety_layers)),
            'end': int(max(safety_layers))
        },

        'parameters': {
            'num_SS_pairs': len(allcos_SS),
            'num_SU_pairs': len(allcos_SU),
            'epsilon': epsilon
        },
        'layer_analysis': layer_analysis
    }

    json_path = os.path.join(save_dir, 'safety_layers.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(safety_layers_result, f, indent=2, ensure_ascii=False)
    print(f"Safety layer localization results saved to: {json_path}")

    summary = {
        'model': model_path.split('/')[-1],
        'total_layers': int(total_layers) if total_layers is not None else None,
        'safety_layers': safety_layers,
        'safety_layer_range': None if len(safety_layers) == 0 else f"Layer {min(safety_layers)} - {max(safety_layers)}"
    }

    summary_path = os.path.join(save_dir, 'safety_layers_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Safety layer summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print(f"Model: {model_path.split('/')[-1]}")
    if total_layers is not None:
        print(f"Total layers: {total_layers}")
        print(f"Safety layer percentage: {len(safety_layers) / total_layers * 100:.1f}%")
    print(f"Safety layers: {safety_layers}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute cosine similarity using saved sentence pairs and locate safety layers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='/path/to/model',
        help='Path to pretrained language model'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='/path/to/results',
        help='Path to directory for saving results'
    )

    parser.add_argument(
        '--pairs_json_dir',
        type=str,
        default=None,
        help='Directory containing sentence pair JSON files (if None, automatically search for safe-safe_pairs.json and safe-unsafe_pairs.json in save_dir)'
    )

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        pairs_json_dir=args.pairs_json_dir
    )
