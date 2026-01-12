"""
evaluate_sac.py - SAC threshold estimator evaluation script

Load trained SAC model and evaluate its performance on test set
"""

import numpy as np
import json
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from tqdm import tqdm

class ResidualBlock(nn.Module):
    """Residual block: helps train deeper networks"""
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + x))

class SelfAttention(nn.Module):
    """
    Self-attention mechanism: allows network to focus on important features
    
    Note: For a single state vector, we use self-attention over feature dimensions.
    Treat the feature vector as a "sequence", where each dimension is a "token".
    """
    def __init__(self, dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, dim) - State feature vector
        
        Returns:
            out: (batch_size, dim) - Enhanced feature vector
        """
        batch_size = x.size(0)
        residual = x  # Save residual connection
        
        # Treat feature vector as sequence (each dimension is a token)
        # x: (batch_size, dim) -> (batch_size, seq_len=1, dim)
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch_size, 1, dim)
        K = self.key(x)
        V = self.value(x)
        
        # Multi-head attention: reshape to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)  # (batch_size, num_heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, -1)  # (batch_size, 1, dim)
        out = self.out(out)  # (batch_size, 1, dim)
        
        # Residual connection and normalization
        out = out.squeeze(1)  # (batch_size, dim)
        out = self.norm(out + residual)
        
        return out

class Actor(nn.Module):
    """
    Actor network (policy network)
    """
    
    def __init__(self, state_dim, action_dim=1, hidden_dims=[1024, 512, 256, 128], 
                 use_attention=True, use_residual=True, dropout=0.1):
        super(Actor, self).__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Feature extraction layer (map input to first hidden layer dimension)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Self-attention layer (optional)
        if use_attention:
            self.attention = SelfAttention(hidden_dims[0], num_heads=8)
        
        # Deep network (with residual connections)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                # If dimensions are same, use residual block
                self.layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                # Otherwise use regular layer
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
        
        # Output mean and log standard deviation
        final_dim = hidden_dims[-1]
        self.mean = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, action_dim)
        )
        self.log_std = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, action_dim)
        )
        
        # Log_std range limits
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        """
        Forward pass
        
        Returns:
            mean: Action mean (unclipped)
            log_std: Log standard deviation
        """
        x = self.feature_extractor(state)
        
        # Apply attention (if enabled)
        if self.use_attention:
            x = self.attention(x)
        
        # Pass through deep network
        for layer in self.layers:
            x = layer(x)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(self, state, threshold_min, threshold_max):
        """
        Deterministically get action (for evaluation)
        
        Returns action corresponding to mean
        """
        mean, _ = self.forward(state)
        action_tanh = torch.tanh(mean)
        action = threshold_min + (threshold_max - threshold_min) * (action_tanh + 1) / 2
        return action

class SACEvaluator:
    """SAC model evaluator"""
    
    def __init__(self, model_path: str, device='cuda', threshold_min=-0.1, threshold_max=0.5, hidden_dims=None, dropout=0.1):
        """
        Initialize evaluator
        
        Args:
            model_path: Model file path (.pth file or directory)
            device: Computing device
            threshold_min: Minimum threshold value
            threshold_max: Maximum threshold value
            hidden_dims: Hidden layer dimensions (if None, will be inferred from model)
            dropout: Dropout rate
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.actor = None
        self.load_model()
    
    def load_model(self):
        """Load SAC model (Actor network)"""
        print(f"Loading SAC model: {self.model_path}")
        
        # Determine model file path
        if os.path.isdir(self.model_path):
            best_model_file = os.path.join(self.model_path, 'best_model.pth')
            default_model_file = os.path.join(self.model_path, 'sac_model.pth')
            
            if os.path.exists(best_model_file):
                model_file = best_model_file
                print(f"  Found best model: best_model.pth")
            elif os.path.exists(default_model_file):
                model_file = default_model_file
                print(f"  Using final model: sac_model.pth")
            else:
                raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        else:
            model_file = self.model_path
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file does not exist: {model_file}")
        
        # Load model
        checkpoint = torch.load(model_file, map_location=self.device)
        
        # Infer network parameters from checkpoint
        actor_state_dict = checkpoint['actor_state_dict']
        
        # Check for network features
        has_attention = 'attention.query.weight' in actor_state_dict
        has_feature_extractor = 'feature_extractor.0.weight' in actor_state_dict
        has_residual = any('layers' in key and 'block.0.weight' in key for key in actor_state_dict.keys())
        
        # Infer state_dim
        if 'feature_extractor.0.weight' in actor_state_dict:
            state_dim = actor_state_dict['feature_extractor.0.weight'].shape[1]
        else:
            # Fallback: infer from first layer weight
            first_key = list(actor_state_dict.keys())[0]
            state_dim = actor_state_dict[first_key].shape[1] if len(actor_state_dict[first_key].shape) > 1 else 392
        
        # Infer hidden_dims
        hidden_dims = self.hidden_dims
        if hidden_dims is None:
            if 'feature_extractor.0.weight' in actor_state_dict:
                first_hidden = actor_state_dict['feature_extractor.0.weight'].shape[0]
                # Try to infer other layers
                hidden_dims = [first_hidden]
                # Check dimensions in layers
                layer_keys = [k for k in actor_state_dict.keys() if 'layers' in k and 'weight' in k]
                for key in sorted(layer_keys):
                    if 'layers' in key and 'block' not in key:
                        dim = actor_state_dict[key].shape[0]
                        if dim not in hidden_dims:
                            hidden_dims.append(dim)
                # If inference fails, use default
                if len(hidden_dims) < 2:
                    hidden_dims = [1024, 512, 256, 128]
        
        # Create Actor network
        use_attention = has_attention
        use_residual = has_residual
        self.actor = Actor(
            state_dim,
            hidden_dims=hidden_dims if hidden_dims else [1024, 512, 256, 128],
            use_attention=use_attention,
            use_residual=use_residual,
            dropout=self.dropout
        ).to(self.device)
        
        # Load weights
        self.actor.load_state_dict(actor_state_dict)
        self.actor.eval()
        
        print(f"Successfully loaded SAC model")
        print(f"  State dimension: {state_dim}")
        print(f"  Action space: continuous [{self.threshold_min}, {self.threshold_max}]")
        print(f"  Device: {self.device}")
        print(f"  Network type: Residual + Attention")
        
        # Display model training information
        if 'episode' in checkpoint and checkpoint['episode']:
            print(f"  Training Episode: {checkpoint['episode']}")
        if 'val_f1' in checkpoint and checkpoint['val_f1']:
            print(f"  Validation F1: {checkpoint['val_f1']:.4f}")
        if 'lambda_temp' in checkpoint:
            print(f"  Final Lambda: {checkpoint['lambda_temp']:.4f}")
    
    def extract_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """Extract features from sample"""
        feature_fields = [
            'prompt_embedding',
            'safe_reference_embedding',
            'unsafe_reference_embedding',
            'is_english',
            'text_length',
            'Perplexity',
            'log_prob_mean',
            'log_prob_var',
            'sentiment_positive',
            'sentiment_negative',
            'hurtlex_coverage'
        ]
        
        feature_vector = []
        for field in feature_fields:
            value = sample.get(field)
            if value is None:
                raise ValueError(f"Sample missing feature field: {field}")
            
            if isinstance(value, list):
                feature_vector.extend(value)
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector, dtype=np.float32)
    
    def predict_threshold(self, features: np.ndarray) -> float:
        """
        Predict threshold for single sample
        
        Args:
            features: Feature vector (feature_dim,)
        
        Returns:
            threshold: Predicted threshold (float)
        """
        state_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            threshold_t = self.actor.get_action(state_t, self.threshold_min, self.threshold_max)
        
        threshold = threshold_t.cpu().numpy()[0][0]
        return float(np.clip(threshold, self.threshold_min, self.threshold_max))
    
    def predict_batch(self, features_batch: np.ndarray, batch_size=128) -> np.ndarray:
        """
        Batch predict thresholds (improve efficiency)
        
        Args:
            features_batch: Feature matrix (n_samples, feature_dim)
            batch_size: Batch size
        
        Returns:
            thresholds: Predicted threshold array (n_samples,)
        """
        n_samples = len(features_batch)
        thresholds = []
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch = features_batch[i:i+batch_size]
            states_t = torch.FloatTensor(batch).to(self.device)
            
            with torch.no_grad():
                thresholds_t = self.actor.get_action(states_t, self.threshold_min, self.threshold_max)
            
            batch_thresholds = thresholds_t.cpu().numpy().squeeze()
            if batch_thresholds.ndim == 0:  # Single sample
                batch_thresholds = np.array([batch_thresholds])
            thresholds.extend(batch_thresholds)
        
        thresholds = np.array(thresholds)
        return np.clip(thresholds, self.threshold_min, self.threshold_max)
    
    def evaluate_dataset(self, test_data_path: str, output_path: str = None, 
                        use_batch=True, analyze_errors=True, batch_size=128):
        """
        Evaluate entire test dataset
        
        Args:
            test_data_path: Test data path (JSON file)
            output_path: Output result path (optional)
            use_batch: Whether to use batch inference (faster)
            analyze_errors: Whether to analyze error samples
            batch_size: Batch size for prediction
        
        Returns:
            metrics: Evaluation metrics dictionary
            results: Detailed results list
        """
        print(f"\nEvaluating test set: {test_data_path}")
        
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Number of test samples: {len(test_data)}")
        
        # Batch extract features
        print("Extracting features...")
        features_list = []
        scores = []
        labels = []
        prompts = []
        
        for sample in tqdm(test_data, desc="Extracting features", ncols=80):
            features = self.extract_features(sample)
            features_list.append(features)
            scores.append(sample['score'])
            labels.append(sample['label'])
            prompts.append(sample.get('prompt', ''))
        
        features_array = np.array(features_list)
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Batch predict thresholds
        print("Predicting thresholds...")
        if use_batch:
            predicted_thresholds = self.predict_batch(features_array, batch_size=batch_size)
        else:
            predicted_thresholds = np.array([self.predict_threshold(f) for f in tqdm(features_list, desc="Predicting", ncols=80)])
        
        # Classify using predicted thresholds
        predicted_labels = (scores > predicted_thresholds).astype(int)
        
        # Compute metrics
        correct = (predicted_labels == labels)
        accuracy = correct.sum() / len(labels)
        
        tp = ((predicted_labels == 1) & (labels == 1)).sum()
        fp = ((predicted_labels == 1) & (labels == 0)).sum()
        tn = ((predicted_labels == 0) & (labels == 0)).sum()
        fn = ((predicted_labels == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compare with fixed threshold 0.25
        fixed_threshold = 0.25
        fixed_predictions = (scores > fixed_threshold).astype(int)
        fixed_correct = (fixed_predictions == labels)
        fixed_accuracy = fixed_correct.sum() / len(labels)
        fixed_tp = ((fixed_predictions == 1) & (labels == 1)).sum()
        fixed_fp = ((fixed_predictions == 1) & (labels == 0)).sum()
        fixed_tn = ((fixed_predictions == 0) & (labels == 0)).sum()
        fixed_fn = ((fixed_predictions == 0) & (labels == 1)).sum()
        fixed_precision = fixed_tp / (fixed_tp + fixed_fp) if (fixed_tp + fixed_fp) > 0 else 0
        fixed_recall = fixed_tp / (fixed_tp + fixed_fn) if (fixed_tp + fixed_fn) > 0 else 0
        fixed_f1 = 2 * fixed_precision * fixed_recall / (fixed_precision + fixed_recall) if (fixed_precision + fixed_recall) > 0 else 0
        
        # Build results
        results = []
        for i in range(len(test_data)):
            result = {
                'predicted_threshold': round(float(predicted_thresholds[i]), 4),
                'predicted_label': int(predicted_labels[i]),
                'score': round(float(scores[i]), 4),
                'label': int(labels[i]),
                'correct': bool(correct[i]),
                'prompt': prompts[i],
                'error_type': self._get_error_type(predicted_labels[i], labels[i])
            }
            results.append(result)
        
        # Statistics by score bins
        score_bins = [(0.0, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.4), (0.4, 1.0)]
        score_bin_stats = {}
        for bin_start, bin_end in score_bins:
            mask = (scores >= bin_start) & (scores < bin_end)
            if mask.sum() > 0:
                bin_accuracy = correct[mask].sum() / mask.sum()
                score_bin_stats[f'{bin_start}-{bin_end}'] = {
                    'count': int(mask.sum()),
                    'accuracy': round(float(bin_accuracy), 4),
                    'avg_threshold': round(float(predicted_thresholds[mask].mean()), 4)
                }
        
        metrics = {
            'total_samples': len(labels),
            'correct': int(correct.sum()),
            'accuracy': round(float(accuracy), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1_score': round(float(f1_score), 4),
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            },
            'threshold_stats': {
                'mean': round(float(predicted_thresholds.mean()), 4),
                'std': round(float(predicted_thresholds.std()), 4),
                'min': round(float(predicted_thresholds.min()), 4),
                'max': round(float(predicted_thresholds.max()), 4),
                'median': round(float(np.median(predicted_thresholds)), 4)
            },
            'score_bin_stats': score_bin_stats,
            'vs_fixed_threshold': {
                'fixed_threshold': fixed_threshold,
                'fixed_accuracy': round(float(fixed_accuracy), 4),
                'fixed_f1': round(float(fixed_f1), 4),
                'improvement': round(float(f1_score - fixed_f1), 4),
                'improvement_pct': round(float((f1_score - fixed_f1) / fixed_f1 * 100), 2) if fixed_f1 > 0 else 0
            }
        }
        
        # Print results
        self._print_metrics(metrics)
        
        # Error analysis
        if analyze_errors:
            error_analysis = self._analyze_errors(results, scores, labels, predicted_thresholds)
            metrics['error_analysis'] = error_analysis
        
        # Save results
        if output_path:
            output_data = {
                'metrics': metrics,
                'model_path': self.model_path,
                'test_data_path': test_data_path,
                'algorithm': 'SAC',
                'results': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nEvaluation results saved to: {output_path}")
        
        return metrics, results
    
    def _get_error_type(self, pred, true):
        """Get error type"""
        if pred == true:
            return 'correct'
        elif true == 1 and pred == 0:
            return 'false_negative'  # Missed detection
        else:
            return 'false_positive'  # False alarm
    
    def _print_metrics(self, metrics):
        """Print evaluation metrics"""
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Correct: {metrics['correct']}")
        
        print(f"\nClassification Metrics:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1 Score:  {metrics['f1_score']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  - True Positive (TP):  {metrics['confusion_matrix']['tp']} (Correctly identified harmful)")
        print(f"  - False Positive (FP): {metrics['confusion_matrix']['fp']} (False alarm)")
        print(f"  - True Negative (TN):  {metrics['confusion_matrix']['tn']} (Correctly identified safe)")
        print(f"  - False Negative (FN): {metrics['confusion_matrix']['fn']} (Missed detection)")
        
        print(f"\nPredicted Threshold Statistics:")
        print(f"  - Mean:   {metrics['threshold_stats']['mean']:.4f}")
        print(f"  - Median: {metrics['threshold_stats']['median']:.4f}")
        print(f"  - Std:    {metrics['threshold_stats']['std']:.4f}")
        print(f"  - Range:  [{metrics['threshold_stats']['min']:.4f}, {metrics['threshold_stats']['max']:.4f}]")
        
        if 'score_bin_stats' in metrics and metrics['score_bin_stats']:
            print(f"\nStatistics by Score Bins:")
            for bin_range, stats in metrics['score_bin_stats'].items():
                print(f"  - [{bin_range}]: {stats['count']} samples, "
                      f"accuracy={stats['accuracy']:.4f}, "
                      f"avg_threshold={stats['avg_threshold']:.4f}")
        
        if 'vs_fixed_threshold' in metrics:
            vs = metrics['vs_fixed_threshold']
            print(f"\nComparison with Fixed Threshold:")
            print(f"  - Fixed threshold (0.25): F1={vs['fixed_f1']:.4f}, Accuracy={vs['fixed_accuracy']:.4f}")
            print(f"  - SAC dynamic threshold:  F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            improvement = vs['improvement']
            if improvement > 0:
                print(f"  - F1 Score improvement: +{improvement:.4f} (+{vs['improvement_pct']:.2f}%)")
            else:
                print(f"  - F1 Score change: {improvement:.4f} ({vs['improvement_pct']:.2f}%)")
        
        print("="*70)
    
    def _analyze_errors(self, results, scores, labels, predicted_thresholds):
        """Analyze error samples"""
        print(f"\nError Analysis:")
        
        # Separate error samples
        errors = [r for r in results if not r['correct']]
        fn_errors = [r for r in errors if r['error_type'] == 'false_negative']
        fp_errors = [r for r in errors if r['error_type'] == 'false_positive']
        
        print(f"  Total errors: {len(errors)}")
        print(f"  - False negatives (missed harmful): {len(fn_errors)}")
        print(f"  - False positives (false alarm): {len(fp_errors)}")
        
        error_analysis = {
            'total_errors': len(errors),
            'false_negatives': len(fn_errors),
            'false_positives': len(fp_errors)
        }
        
        # Analyze false negatives
        if fn_errors:
            fn_scores = [e['score'] for e in fn_errors]
            fn_thresholds = [e['predicted_threshold'] for e in fn_errors]
            fn_distances = [abs(e['score'] - e['predicted_threshold']) for e in fn_errors]
            print(f"\n  False Negative Analysis:")
            print(f"    - Score mean: {np.mean(fn_scores):.4f}")
            print(f"    - Threshold mean: {np.mean(fn_thresholds):.4f}")
            print(f"    - Average distance: {np.mean(fn_distances):.4f}")
            error_analysis['fn_stats'] = {
                'avg_score': round(float(np.mean(fn_scores)), 4),
                'avg_threshold': round(float(np.mean(fn_thresholds)), 4),
                'avg_distance': round(float(np.mean(fn_distances)), 4)
            }
        
        # Analyze false positives
        if fp_errors:
            fp_scores = [e['score'] for e in fp_errors]
            fp_thresholds = [e['predicted_threshold'] for e in fp_errors]
            fp_distances = [abs(e['score'] - e['predicted_threshold']) for e in fp_errors]
            print(f"\n  False Positive Analysis:")
            print(f"    - Score mean: {np.mean(fp_scores):.4f}")
            print(f"    - Threshold mean: {np.mean(fp_thresholds):.4f}")
            print(f"    - Average distance: {np.mean(fp_distances):.4f}")
            error_analysis['fp_stats'] = {
                'avg_score': round(float(np.mean(fp_scores)), 4),
                'avg_threshold': round(float(np.mean(fp_thresholds)), 4),
                'avg_distance': round(float(np.mean(fp_distances)), 4)
            }
        
        return error_analysis

def main():
    parser = argparse.ArgumentParser(description='SAC threshold estimator evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Model path (.pth file or directory)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Test data path (directory or JSON file)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Computing device (cuda/cpu), default: auto')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.upper()}")
    
    # Hardcoded parameters (same as train.py)
    THRESHOLD_MIN = -0.1
    THRESHOLD_MAX = 0.5
    HIDDEN_DIMS = [1024, 512, 256, 128]
    DROPOUT = 0.1
    
    evaluator = SACEvaluator(
        args.model_path, 
        device=device,
        threshold_min=THRESHOLD_MIN,
        threshold_max=THRESHOLD_MAX,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    )
    
    # Evaluate test data
    if os.path.isdir(args.test_data):
        # If directory, recursively search all JSON files
        test_files = []
        for root, dirs, files in os.walk(args.test_data):
            for f in files:
                if f.endswith('.json'):
                    test_files.append((os.path.join(root, f), f))
        
        print(f"\nFound {len(test_files)} test files")
        
        if len(test_files) == 0:
            print(f"\nWarning: No JSON test files found!")
            print(f"Please check directory: {args.test_data}")
            return
        
        all_metrics = {}
        for test_path, test_filename in test_files:
            # Create output filename (preserve directory structure info)
            relative_path = os.path.relpath(test_path, args.test_data)
            output_filename = f"eval_{relative_path.replace(os.sep, '_')}"
            output_path = os.path.join(args.output_dir, output_filename)
            
            print(f"\n{'='*60}")
            print(f"Evaluating file: {relative_path}")
            print(f"{'='*60}")
            
            metrics, _ = evaluator.evaluate_dataset(
                test_path, 
                output_path,
                use_batch=True,
                analyze_errors=False,
                batch_size=args.batch_size
            )
            all_metrics[relative_path] = metrics
        
        # Print summary
        print(f"\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        
        # Group by dataset type
        dataset_groups = {}
        for test_file, metrics in all_metrics.items():
            # Extract dataset name (e.g., toxicchat, wildguard, etc.)
            dataset_name = test_file.split(os.sep)[0] if os.sep in test_file else 'unknown'
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append((test_file, metrics))
        
        # Print by dataset
        for dataset_name in sorted(dataset_groups.keys()):
            files_metrics = dataset_groups[dataset_name]
            print(f"\n[{dataset_name.upper()}]")
            
            for test_file, metrics in files_metrics:
                print(f"  {os.path.basename(test_file)}:")
                print(f"    - F1 Score:  {metrics['f1_score']:.4f}")
                print(f"    - Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    - Precision: {metrics['precision']:.4f}")
                print(f"    - Recall:    {metrics['recall']:.4f}")
            
            # Calculate dataset average
            if len(files_metrics) > 1:
                avg_f1 = np.mean([m[1]['f1_score'] for m in files_metrics])
                avg_acc = np.mean([m[1]['accuracy'] for m in files_metrics])
                print(f"  Average:")
                print(f"    - F1 Score:  {avg_f1:.4f}")
                print(f"    - Accuracy:  {avg_acc:.4f}")
        
        # Overall average
        if len(all_metrics) > 1:
            print(f"\n[Overall Average]")
            overall_f1 = np.mean([m['f1_score'] for m in all_metrics.values()])
            overall_acc = np.mean([m['accuracy'] for m in all_metrics.values()])
            print(f"  - F1 Score:  {overall_f1:.4f}")
            print(f"  - Accuracy:  {overall_acc:.4f}")
        
        print(f"{'='*70}")
    
    else:
        # Single file
        output_path = os.path.join(args.output_dir, 'evaluation_results.json')
        evaluator.evaluate_dataset(
            args.test_data, 
            output_path,
            use_batch=True,
            analyze_errors=False,
            batch_size=args.batch_size
        )
    
    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()

