import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import logging
import torch
import torch.nn.functional as F
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure paths
MODEL_PATH = '/path/to/bge-m3'
CSV_FILE = '/path/to/test.csv'
BENIGN_DB_DIR = '/path/to/database/benign_tcwg'
HARMFUL_DB_DIR = '/path/to/database/harmful_tcwg'
OUTPUT_JSON = '/path/to/retrieve_tcwg_3.json'
TOP_K = 3
# Configure number of reference texts (can be modified directly in code, command line arguments will override)
NUM_SAFE_REFERENCES = TOP_K  # Number of safe reference texts retrieved per prompt
NUM_UNSAFE_REFERENCES = TOP_K  # Number of unsafe reference texts retrieved per prompt

# Create output directory
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


class ReferenceRetriever:
    """Class for retrieving reference texts"""
    
    def __init__(self, model_path):
        """Initialize model"""
        logger.info(f"Loading BGE-M3 model from {model_path}...")
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=True,
            device="cuda"
        )
        logger.info("✓ Model loaded successfully")
    
    def load_database(self, db_dir):
        """Load database"""
        logger.info(f"Loading database from {db_dir}...")
        
        # Load texts
        with open(os.path.join(db_dir, 'texts.pkl'), 'rb') as f:
            texts = pickle.load(f)
        
        # Load Dense embeddings, transfer to GPU
        dense_embeddings = np.load(os.path.join(db_dir, 'dense_embeddings.npy'))
        dense_embeddings = torch.tensor(dense_embeddings, dtype=torch.float32, device='cuda')
        
        # Load Multi-vector embeddings - support both old and new formats
        multi_vector_dir = os.path.join(db_dir, 'multi_vector_embeddings')
        if os.path.isdir(multi_vector_dir):
            # New format: chunked multi-vector embeddings
            logger.info("Loading multi-vector embeddings from chunks...")
            multi_vector_embeddings = []
            chunk_files = sorted([f for f in os.listdir(multi_vector_dir) if f.endswith('.pkl')])
            for chunk_file in tqdm(chunk_files, desc="Loading multi-vector chunks"):
                with open(os.path.join(multi_vector_dir, chunk_file), 'rb') as f:
                    chunk = pickle.load(f)
                    multi_vector_embeddings.extend(chunk)
            logger.info(f"✓ Loaded {len(multi_vector_embeddings)} multi-vectors from {len(chunk_files)} chunks")
        else:
            # Old format: single pickle file (existing database)
            logger.info("Loading multi-vector embeddings from single file...")
            with open(os.path.join(db_dir, 'multi_vector_embeddings.pkl'), 'rb') as f:
                multi_vector_embeddings = pickle.load(f)
        
        logger.info(f"✓ Database loaded: {len(texts)} texts")
        return texts, dense_embeddings, multi_vector_embeddings
    
    def count_words(self, text):
        """Count words in text"""
        if not text or not isinstance(text, str):
            return 0
        return len(text.split())
    
    def encode_prompt(self, prompt):
        """Encode single prompt"""
        outputs = self.model.encode(
            [prompt],
            return_dense=True,
            return_colbert_vecs=True
        )
        dense_vec = outputs['dense_vecs'][0]  # (1024,)
        # Convert to GPU tensor
        dense_vec = torch.tensor(dense_vec, dtype=torch.float32, device='cuda')
        multi_vec = outputs['colbert_vecs'][0]  # Multi-vector
        return dense_vec, multi_vec
    
    def rerank_by_multi_vector(self, prompt_multi_vec, candidate_indices, 
                               all_multi_vectors, top_k=100):
        """Rerank candidates using Multi-Vector (ColBERT) (GPU optimized version)
        
        Returns: (reranked_indices, multi_sim_dict)
        - reranked_indices: List of reranked indices
        - multi_sim_dict: {idx: multi_sim} dictionary storing Multi-Vector similarity for each index
        """
        scores = []
        
        # Convert prompt_multi_vec to GPU tensor (if not already)
        if isinstance(prompt_multi_vec, np.ndarray):
            prompt_multi_vec = torch.tensor(prompt_multi_vec, dtype=torch.float32, device='cuda')
        else:
            prompt_multi_vec = prompt_multi_vec.to('cuda')
        
        # Batch process candidates to avoid processing one by one
        batch_size = 100  # Process 100 per batch
        
        for batch_start in range(0, len(candidate_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(candidate_indices))
            batch_indices = candidate_indices[batch_start:batch_end]
            
            for idx in batch_indices:
                # Get candidate vector
                candidate_multi_vec = all_multi_vectors[idx]
                
                if isinstance(candidate_multi_vec, np.ndarray):
                    candidate_multi_vec = torch.tensor(candidate_multi_vec, dtype=torch.float32, device='cuda')
                else:
                    candidate_multi_vec = candidate_multi_vec.to('cuda')
                
                # Compute maximum similarity (MaxSim)
                # prompt_multi_vec: (num_tokens_prompt, dim)
                # candidate_multi_vec: (num_tokens_candidate, dim)
                if len(prompt_multi_vec) == 0 or len(candidate_multi_vec) == 0:
                    scores.append((idx, 0.0))
                    del candidate_multi_vec
                    continue
                
                # Compute similarity for all token pairs on GPU, take maximum
                similarity_matrix = torch.mm(prompt_multi_vec, candidate_multi_vec.t())  # (num_tokens_prompt, num_tokens_candidate)
                max_similarity = torch.max(similarity_matrix).item()  # Extract scalar value
                scores.append((idx, max_similarity))
                
                # Clean GPU memory promptly
                del candidate_multi_vec, similarity_matrix
        
        # Sort by Multi-Vector similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        reranked_indices = [idx for idx, _ in scores[:top_k]]
        
        # Create similarity dictionary to avoid repeated computation
        multi_sim_dict = {idx: sim for idx, sim in scores}
        
        # Clean GPU memory
        del prompt_multi_vec
        torch.cuda.empty_cache()
        
        return reranked_indices, multi_sim_dict
    
    def retrieve_references(self, prompt, db_texts, db_dense_embeddings, 
                           db_multi_vectors, num_references=2):
        """Retrieve reference texts (batch processing: 100 per batch, rerank by multi-vector then find qualified ones)"""
        prompt_word_count = self.count_words(prompt)
        
        # Handle empty prompt or word count 0
        if prompt_word_count == 0:
            logger.warning(f"Empty prompt detected, using default word range")
            min_words = 1
            max_words = 100
        else:
            min_words = max(1, int(prompt_word_count * 0.5))
            max_words = int(prompt_word_count * 2)
        
        # 1. Encode prompt
        prompt_dense_vec, prompt_multi_vec = self.encode_prompt(prompt)
        
        # 2. Compute similarity for all Dense embeddings (compute once)
        # Normalize
        prompt_vec_normalized = F.normalize(prompt_dense_vec.unsqueeze(0), p=2, dim=1)  # (1, 1024)
        db_vecs_normalized = F.normalize(db_dense_embeddings, p=2, dim=1)  # (num_texts, 1024)
        
        # Compute all similarities
        similarities_gpu = torch.mm(prompt_vec_normalized, db_vecs_normalized.t()).squeeze(0)  # (num_texts,)
        
        # Get all indices sorted by similarity (high to low)
        all_scores, all_indices = torch.sort(similarities_gpu, descending=True)
        all_indices = all_indices.cpu().numpy().tolist()
        
        # 3. Batch processing: 100 per batch, rerank by multi-vector within each batch
        candidate_list = []  # Store qualified candidates (idx, dense_sim, multi_sim, text)
        seen_texts = set()  # For deduplication
        batch_size = 100
        
        # Iterate through all candidates, process in batches
        batch_start = 0
        while batch_start < len(all_indices) and len(candidate_list) < num_references:
            batch_end = min(batch_start + batch_size, len(all_indices))
            batch_indices = all_indices[batch_start:batch_end]
            
            # Rerank current batch by Multi-Vector (returns indices and similarity dictionary)
            reranked_indices, multi_sim_dict = self.rerank_by_multi_vector(
                prompt_multi_vec, batch_indices, db_multi_vectors, top_k=len(batch_indices)
            )
            
            # Within reranked batch, check from high to low by multi-vector similarity
            for idx in reranked_indices:
                # If enough candidates found, stop
                if len(candidate_list) >= num_references:
                    break
                
                text = db_texts[idx]
                
                # Deduplication check: ensure no duplicates
                if text in seen_texts:
                    continue
                
                # Get Dense similarity
                dense_sim = similarities_gpu[idx].item()
                
                # Check word count
                text_word_count = self.count_words(text)
                if not (min_words <= text_word_count <= max_words):
                    continue
                
                # Use already computed Multi-Vector similarity
                multi_sim = multi_sim_dict[idx]
                
                # All checks passed, add to candidate list
                candidate_list.append((idx, dense_sim, multi_sim, text))
                seen_texts.add(text)
            
            # If enough candidates found, exit early
            if len(candidate_list) >= num_references:
                break
            
            # Continue to next batch
            batch_start = batch_end
        
        # 4. Check if enough candidates found
        if len(candidate_list) == 0:
            logger.warning("No qualified candidates found")
            # Clean GPU memory
            del prompt_dense_vec, prompt_multi_vec, similarities_gpu, prompt_vec_normalized, db_vecs_normalized
            torch.cuda.empty_cache()
            return []
        
        # 5. Return selected reference texts (already sorted by multi-vector similarity high to low)
        selected_references = [text for _, _, _, text in candidate_list[:num_references]]
        
        # Clean GPU memory
        del prompt_dense_vec, prompt_multi_vec, similarities_gpu, prompt_vec_normalized, db_vecs_normalized
        torch.cuda.empty_cache()
        
        return selected_references
    
    def process_toxicchat(self, num_safe_references=2, num_unsafe_references=2):
        """Process ToxicChat dataset
        
        Args:
            num_safe_references: Number of safe reference texts retrieved per prompt, default 2
            num_unsafe_references: Number of unsafe reference texts retrieved per prompt, default 2
        """
        logger.info(f"\n{'='*90}")
        logger.info("Processing ToxicChat dataset")
        logger.info(f"  • Safe references per prompt: {num_safe_references}")
        logger.info(f"  • Unsafe references per prompt: {num_unsafe_references}")
        logger.info(f"{'='*90}\n")
        
        # 2. Read CSV file
        logger.info(f"Reading CSV from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        logger.info(f"✓ Loaded {len(df)} prompts")
        
        # 3. First stage: process all safe_reference
        logger.info(f"\n{'='*90}")
        logger.info("Stage 1: Retrieving safe reference texts (Safe References)")
        logger.info(f"{'='*90}\n")
        
        # Load benign database
        logger.info("Loading benign database...")
        benign_texts, benign_dense, benign_multi = self.load_database(BENIGN_DB_DIR)
        
        # Store results
        results = []
        
        # Process safe_reference for each prompt
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving safe references"):
            result_item = {
                'conv_id': str(row['conv_id']) if pd.notna(row['conv_id']) else '',
                'user_input': str(row['user_input']) if pd.notna(row['user_input']) else '',
                'model_output': str(row['model_output']) if pd.notna(row['model_output']) else '',
                'human_annotation': str(row['human_annotation']) if pd.notna(row['human_annotation']) else '',
                'toxicity': str(row['toxicity']) if pd.notna(row['toxicity']) else '',
                'jailbreaking': str(row['jailbreaking']) if pd.notna(row['jailbreaking']) else '',
                'openai_moderation': str(row['openai_moderation']) if pd.notna(row['openai_moderation']) else '',
            }
            
            # Use user_input field as query text (avoid NaN)
            prompt = result_item['user_input']
            
            # Retrieve safe reference texts
            try:
                safe_references = self.retrieve_references(
                    prompt, benign_texts, benign_dense, benign_multi, num_references=num_safe_references
                )
                result_item['safe_reference'] = safe_references if safe_references else []
            except Exception as e:
                logger.exception(f"Error retrieving safe references for item {idx}")
                result_item['safe_reference'] = []
            
            results.append(result_item)
        
        # Clean benign database, release GPU memory
        logger.info("\nCleaning up benign database...")
        del benign_texts, benign_dense, benign_multi
        torch.cuda.empty_cache()
        logger.info("✓ Benign database cleaned from GPU")
        
        # 4. Second stage: process all unsafe_reference
        logger.info(f"\n{'='*90}")
        logger.info("Stage 2: Retrieving unsafe reference texts (Unsafe References)")
        logger.info(f"{'='*90}\n")
        
        # Load harmful database
        logger.info("Loading harmful database...")
        harmful_texts, harmful_dense, harmful_multi = self.load_database(HARMFUL_DB_DIR)
        
        # Process unsafe_reference for each prompt
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving unsafe references"):
            # Use already converted user_input field as query text (avoid NaN)
            prompt = results[idx]['user_input']
            
            # Retrieve unsafe reference texts
            try:
                unsafe_references = self.retrieve_references(
                    prompt, harmful_texts, harmful_dense, harmful_multi, num_references=num_unsafe_references
                )
                results[idx]['unsafe_reference'] = unsafe_references if unsafe_references else []
            except Exception as e:
                logger.exception(f"Error retrieving unsafe references for item {idx}")
                results[idx]['unsafe_reference'] = []
        
        # Clean harmful database
        logger.info("\nCleaning up harmful database...")
        del harmful_texts, harmful_dense, harmful_multi
        torch.cuda.empty_cache()
        logger.info("✓ Harmful database cleaned from GPU")
        
        # 5. Save results
        logger.info(f"\n{'='*90}")
        logger.info("Saving results")
        logger.info(f"{'='*90}\n")
        logger.info(f"Saving results to {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Results saved")
        
        logger.info(f"\n{'='*90}")
        logger.info("✅ Processing completed successfully!")
        logger.info(f"{'='*90}\n")
        
        # Statistics
        safe_count = sum(1 for r in results if len(r.get('safe_reference', [])) >= num_safe_references)
        unsafe_count = sum(1 for r in results if len(r.get('unsafe_reference', [])) >= num_unsafe_references)
        logger.info(f"📊 Statistics:")
        logger.info(f"  • Total prompts: {len(results)}")
        logger.info(f"  • Prompts with {num_safe_references} safe references: {safe_count}")
        logger.info(f"  • Prompts with {num_unsafe_references} unsafe references: {unsafe_count}")


def main():
    """Main function"""
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='reference text retrieval')
    parser.add_argument('--num_safe_references', type=int, default=None,
                        help=f'Number of safe reference texts retrieved per prompt (default: {NUM_SAFE_REFERENCES}, can be modified in code)')
    parser.add_argument('--num_unsafe_references', type=int, default=None,
                        help=f'Number of unsafe reference texts retrieved per prompt (default: {NUM_UNSAFE_REFERENCES}, can be modified in code)')
    
    args = parser.parse_args()
    
    # Parameter priority: command line arguments > code configuration
    num_safe_refs = args.num_safe_references if args.num_safe_references is not None else NUM_SAFE_REFERENCES
    num_unsafe_refs = args.num_unsafe_references if args.num_unsafe_references is not None else NUM_UNSAFE_REFERENCES
    
    retriever = ReferenceRetriever(MODEL_PATH)
    retriever.process_toxicchat(
        num_safe_references=num_safe_refs,
        num_unsafe_references=num_unsafe_refs
    )


if __name__ == '__main__':
    main()
