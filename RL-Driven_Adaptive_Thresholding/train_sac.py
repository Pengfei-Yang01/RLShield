"""
train_sac.py - SAC (Soft Actor-Critic) based threshold estimator training script



Uses SAC (Soft Actor-Critic) algorithm

Core advantages:
- Continuous action space: Directly output precise threshold (e.g., 0.237), not discrete categories
- Automatic exploration: Balance exploration and exploitation through entropy regularization
- High sample efficiency: Off-policy algorithm, can reuse experience
- Stable training: Uses two Critic networks, takes minimum to prevent overestimation

SAC core components:
1. Actor network (policy network): Outputs mean and standard deviation of actions
2. Two Critic networks: Estimate Q-values, take minimum to prevent overestimation
3. Target Critic networks: Provide stable training targets
4. Automatic temperature tuning: Dynamically adjust exploration degree
"""

import numpy as np
import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from sklearn.model_selection import train_test_split

# ===========================
# Set random seed
# ===========================
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ===========================
# Threshold configuration
# ===========================
THRESHOLD_MIN = -0.1  # Minimum threshold
THRESHOLD_MAX = 0.5  # Maximum threshold

class ThresholdEnvironment:
    """
    Threshold selection environment (continuous action version)
    
    Each sample is an episode:
    - Environment provides state (feature vector)
    - Agent outputs continuous threshold
    - Environment returns reward (based on classification correctness)
    """
    
    def __init__(self, features, labels, scores):
        """
        Initialize environment
        
        Args:
            features: Feature matrix (n_samples, feature_dim)
            labels: Label array (n_samples,)
            scores: Pre-computed score array (n_samples,)
        """
        self.features = features
        self.labels = labels
        self.scores = scores
        self.n_samples = len(labels)
        self.current_idx = 0
        
        # Randomly shuffle sample order
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]
        self.scores = self.scores[indices]
    
    def reset(self):
        """Reset environment, return state of new sample"""
        self.current_idx = (self.current_idx + 1) % self.n_samples
        state = self.features[self.current_idx]
        return state
    
    def step(self, threshold):
        """
        Execute action, return reward
        
        Args:
            threshold: Continuous threshold (float)
        
        Returns:
            next_state: State of next sample
            reward: Reward value
            done: Whether episode is done
            info: Additional information
        """
        # Get current sample information
        true_label = self.labels[self.current_idx]
        score = self.scores[self.current_idx]
        
        # Clip threshold to valid range
        threshold = np.clip(threshold, THRESHOLD_MIN, THRESHOLD_MAX)
        
        # Classification rule: score > threshold → unsafe(1), otherwise → safe(0)
        predicted_label = 1 if score > threshold else 0
        
        # Compute distance
        distance = abs(threshold - score)
        
        # Compute base reward (continuous reward + differentiated penalty)
        if predicted_label == true_label:
            base_reward = min(10 * distance, 1.5)
        else:
            base_reward = -1 - 5 * distance
        
        reward = base_reward
        
        # End current episode, get next state
        done = True
        next_state = self.reset()
        
        info = {
            'predicted_label': predicted_label,
            'true_label': true_label,
            'threshold': threshold,
            'score': score,
            'correct': predicted_label == true_label
        }
        
        return next_state, reward, done, info

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

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
    
    Input: State features
    Output: Mean and log standard deviation of actions
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
    
    def sample(self, state):
        """
        Sample action (with reparameterization)
        
        Returns:
            action: Sampled action (clipped to [threshold_min, threshold_max])
            log_prob: Log probability
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization sampling
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Use rsample() to support gradient backpropagation
        
        # Use tanh to clip to (-1, 1), then map to [threshold_min, threshold_max]
        action_tanh = torch.tanh(x_t)
        action = THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * (action_tanh + 1) / 2
        
        # Compute log probability (considering tanh transformation Jacobian)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)  # Tanh Jacobian
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state):
        """
        Deterministically get action (for evaluation)
        
        Returns action corresponding to mean
        """
        mean, _ = self.forward(state)
        action_tanh = torch.tanh(mean)
        action = THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * (action_tanh + 1) / 2
        return action

class Critic(nn.Module):
    """
    Critic network (Dueling architecture)
    
    Dueling architecture decomposes Q-value as:
    Q(s,a) = V(s) + A(s,a)
    
    where:
    - V(s): State value (how good is this state)
    - A(s,a): Advantage function (how much better is this action than average)
    
    Advantages:
    1. Better learning of state values
    2. More stable training
    3. Better generalization
    """
    
    def __init__(self, state_dim, action_dim=1, hidden_dims=[1024, 512, 256, 128],
                 use_residual=True, dropout=0.1):
        super(Critic, self).__init__()
        
        self.use_residual = use_residual
        
        # State feature extraction
        self.state_layers = nn.ModuleList()
        state_input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            if use_residual and state_input_dim == hidden_dim:
                self.state_layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                self.state_layers.append(nn.Sequential(
                    nn.Linear(state_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
                state_input_dim = hidden_dim
        
        # Action feature extraction
        self.action_layers = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fuse state and action
        fusion_dim = hidden_dims[-1] + hidden_dims[-1] // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Dueling architecture: separate state value and advantage
        # State value branch V(s)
        self.value_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Advantage branch A(s,a)
        self.advantage_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
    
    def forward(self, state, action):
        """
        Forward pass
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
        
        Returns:
            q_value: (batch_size, 1)
        """
        # Extract state features
        state_feat = state
        for layer in self.state_layers:
            state_feat = layer(state_feat)
        
        # Extract action features
        action_feat = self.action_layers(action)
        
        # Fuse state and action
        combined = torch.cat([state_feat, action_feat], dim=1)
        fused = self.fusion(combined)
        
        # Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = self.value_branch(fused)
        advantage = self.advantage_branch(fused)
        
        # Center advantage (subtract mean, improve stability)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_value

class SACAgent:
    """
    SAC Agent
    
    Core features:
    1. Actor-Critic architecture
    2. Two Critic networks (prevent overestimation)
    3. Automatic temperature tuning (control exploration)
    4. Soft update target networks
    """
    
    def __init__(self, state_dim, device='cuda', learning_rate=3e-4,
                 gamma=0.99, tau=0.005, lambda_temp=0.2, automatic_entropy_tuning=True,
                 hidden_dims=[1024, 512, 256, 128], dropout=0.1):
        """
        Initialize SAC Agent
        
        Args:
            state_dim: State dimension
            device: Computing device
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            lambda_temp: Temperature parameter (controls entropy weight)
            automatic_entropy_tuning: Whether to automatically tune temperature
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.lambda_temp = lambda_temp
        self.hidden_dims = hidden_dims
        
        # Create networks
        self.actor = Actor(
            state_dim, 
            hidden_dims=hidden_dims,
            use_attention=True,
            use_residual=True,
            dropout=dropout
        ).to(self.device)
        self.critic1 = Critic(
            state_dim,
            hidden_dims=hidden_dims,
            use_residual=True,
            dropout=dropout
        ).to(self.device)
        self.critic2 = Critic(
            state_dim,
            hidden_dims=hidden_dims,
            use_residual=True,
            dropout=dropout
        ).to(self.device)
        self.critic1_target = Critic(
            state_dim,
            hidden_dims=hidden_dims,
            use_residual=True,
            dropout=dropout
        ).to(self.device)
        self.critic2_target = Critic(
            state_dim,
            hidden_dims=hidden_dims,
            use_residual=True,
            dropout=dropout
        ).to(self.device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Automatic temperature tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -1  # Target entropy (negative of action_dim)
            self.log_lambda = torch.zeros(1, requires_grad=True, device=self.device)
            self.lambda_optimizer = optim.Adam([self.log_lambda], lr=learning_rate)
            self.lambda_temp = self.log_lambda.exp()
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        
        # Training records
        self.train_losses = []
        self.episode_rewards = []
        self.episode_accuracies = []
        self.lambda_values = []
    
    def select_action(self, state, deterministic=False):
        """
        Select action
        
        Args:
            state: State (numpy array)
            deterministic: Whether to use deterministic policy (for evaluation)
        
        Returns:
            action: Action value (float)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_action(state_t)
            else:
                action, _ = self.actor.sample(state_t)
        self.actor.train()
        
        return action.cpu().numpy()[0][0]
    
    def update(self, batch_size=256):
        """
        Update SAC networks
        
        SAC update steps:
        1. Update two Critic networks
        2. Update Actor network
        3. Update temperature parameter (if automatic tuning)
        4. Soft update target networks
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ===========================
        # Update Critic
        # ===========================
        with torch.no_grad():
            # Sample next action
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Compute target Q-value (take minimum of two Critics)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.lambda_temp * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update Critic1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update Critic2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ===========================
        # Update Actor
        # ===========================
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor loss (maximize Q - lambda * log_prob)
        actor_loss = (self.lambda_temp * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===========================
        # Update temperature parameter
        # ===========================
        if self.automatic_entropy_tuning:
            lambda_loss = -(self.log_lambda * (log_probs + self.target_entropy).detach()).mean()
            
            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()
            
            self.lambda_temp = self.log_lambda.exp()
            self.lambda_values.append(self.lambda_temp.item())
        
        # ===========================
        # Soft update target networks
        # ===========================
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)
        
        # Record loss
        total_loss = critic1_loss.item() + critic2_loss.item() + actor_loss.item()
        return total_loss
    
    def soft_update(self, target, source):
        """
        Soft update target network
        
        target = τ * source + (1 - τ) * target
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

def load_data(file_path):
    """Load training data"""
    print(f"Loading data: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features_list = []
    labels_list = []
    scores_list = []
    
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
    
    for item in data:
        label = item['label']
        score = item['score']
        
        feature_vector = []
        for field in feature_fields:
            value = item[field]
            if isinstance(value, list):
                feature_vector.extend(value)
            else:
                feature_vector.append(value)
        
        features_list.append(feature_vector)
        labels_list.append(label)
        scores_list.append(score)
    
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    scores = np.array(scores_list, dtype=np.float32)
    
    print(f"Loaded: {len(labels)} samples, feature dimension={features.shape[1]}")
    
    return features, labels, scores

def evaluate_agent(agent, env, n_episodes=100):
    """Evaluate agent performance"""
    agent.actor.eval()
    
    correct = 0
    total_reward = 0.0
    
    tp = fp = tn = fn = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state, deterministic=True)
        _, reward, _, info = env.step(action)
        
        pred_label = info['predicted_label']
        true_label = info['true_label']
        
        if pred_label == true_label:
            correct += 1
        
        total_reward += reward
        
        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1
        elif pred_label == 0 and true_label == 1:
            fn += 1
    
    agent.actor.train()
    
    accuracy = correct / n_episodes
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_reward = total_reward / n_episodes
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_reward': avg_reward,
        'correct': correct,
        'total': n_episodes,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def save_agent(agent, save_dir, episode=None, val_f1=None, is_best=False):
    """Save agent model"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'sac_model.pth')
    checkpoint = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic1_state_dict': agent.critic1.state_dict(),
        'critic2_state_dict': agent.critic2.state_dict(),
        'critic1_target_state_dict': agent.critic1_target.state_dict(),
        'critic2_target_state_dict': agent.critic2_target.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict': agent.critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict': agent.critic2_optimizer.state_dict(),
        'lambda_temp': agent.lambda_temp.item() if torch.is_tensor(agent.lambda_temp) else agent.lambda_temp,
        'episode': episode,
        'val_f1': val_f1,
        'train_losses': agent.train_losses,
        'episode_rewards': agent.episode_rewards,
        'episode_accuracies': agent.episode_accuracies
    }
    
    if agent.automatic_entropy_tuning:
        checkpoint['log_lambda'] = agent.log_lambda.item()
        checkpoint['lambda_optimizer_state_dict'] = agent.lambda_optimizer.state_dict()
    
    torch.save(checkpoint, model_path)
    print(f"  Model saved to: {model_path}")
    
    if is_best:
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        print(f"  Best model saved to: {best_model_path}")
        
        best_info_path = os.path.join(save_dir, 'best_model_info.json')
        with open(best_info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'episode': episode,
                'val_f1': float(val_f1) if val_f1 else None,
                'metric_name': 'f1_score',
                'algorithm': 'SAC',
                'timestamp': str(np.datetime64('now'))
            }, f, indent=2, ensure_ascii=False)

def save_checkpoint(agent, save_dir, episode, val_metrics=None):
    """
    Save checkpoint for specified episode
    
    Args:
        agent: SACAgent instance
        save_dir: Save directory
        episode: Episode number
        val_metrics: Validation metrics (optional)
    """
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_episode_{episode}.pth')
    
    checkpoint = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic1_state_dict': agent.critic1.state_dict(),
        'critic2_state_dict': agent.critic2.state_dict(),
        'critic1_target_state_dict': agent.critic1_target.state_dict(),
        'critic2_target_state_dict': agent.critic2_target.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict': agent.critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict': agent.critic2_optimizer.state_dict(),
        'lambda_temp': agent.lambda_temp.item() if torch.is_tensor(agent.lambda_temp) else agent.lambda_temp,
        'episode': episode,
        'train_losses': agent.train_losses,
        'episode_rewards': agent.episode_rewards,
        'episode_accuracies': agent.episode_accuracies
    }
    
    if agent.automatic_entropy_tuning:
        checkpoint['log_lambda'] = agent.log_lambda.item()
        checkpoint['lambda_optimizer_state_dict'] = agent.lambda_optimizer.state_dict()
    
    if val_metrics:
        checkpoint['val_f1'] = val_metrics.get('f1_score', None)
        checkpoint['val_accuracy'] = val_metrics.get('accuracy', None)
        checkpoint['val_precision'] = val_metrics.get('precision', None)
        checkpoint['val_recall'] = val_metrics.get('recall', None)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='SAC threshold estimator training')
    
    # Data paths
    parser.add_argument('--train_data', type=str, nargs='+', required=True, help='Path(s) to training data JSON file(s) (can specify multiple files)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models')
    
    # Device
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu), default: auto')
    
    # Hyperparameters
    parser.add_argument('--n_episodes', type=int, required=True, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--gamma', type=float, required=True, help='Discount factor')
    parser.add_argument('--tau', type=float, required=True, help='Soft update coefficient')
    parser.add_argument('--lambda_temp', type=float, default=None, help='Initial temperature parameter (lambda)')
    
    # Training configuration
    parser.add_argument('--eval_freq', type=int, required=True, help='Evaluation frequency (episodes)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for train_test_split')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("SAC (Soft Actor-Critic) Threshold Estimator Training")
    print("="*60)
    print(f"\nDevice: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nNetwork architecture: Residual + Attention + Dueling Critic")
    
    # Load data
    print("\n[Step 1] Loading dataset...")
    features_list = []
    labels_list = []
    scores_list = []
    
    for data_path in args.train_data:
        print(f"Loading: {data_path}")
        f, l, s = load_data(data_path)
        features_list.append(f)
        labels_list.append(l)
        scores_list.append(s)
    
    # Merge all datasets
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    scores = np.concatenate(scores_list)
    
    print(f"\nTotal samples: {len(labels)} (from {len(args.train_data)} file(s))")
    print(f"Feature dimension: {features.shape[1]}")
    
    X_train, X_val, y_train, y_val, score_train, score_val = train_test_split(
        features, labels, scores,
        test_size=0.2,
        random_state=args.random_seed
    )

    print(f"Training set: {len(y_train)} samples")
    print(f"Validation set: {len(y_val)} samples")
    
    # Create environment and Agent
    print("\n[Step 2] Creating environment and Agent...")
    train_env = ThresholdEnvironment(X_train, y_train, score_train)
    val_env = ThresholdEnvironment(X_val, y_val, score_val)
    
    state_dim = X_train.shape[1]
    
    lambda_temp = args.lambda_temp if args.lambda_temp is not None else 0.2
    
    agent = SACAgent(
        state_dim=state_dim,
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        lambda_temp=lambda_temp,
        automatic_entropy_tuning=True,
        hidden_dims=[1024, 512, 256, 128],
        dropout=0.1
    )
    
    print(f"✓ SAC Agent created")
    print(f"  State dimension: {state_dim}")
    print(f"  Action space: continuous [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print(f"  Automatic temperature tuning: True")
    print(f"  Network architecture: Residual + Attention + Dueling Critic")
    print(f"    - Actor: [1024, 512, 256, 128] + residual + attention")
    print(f"    - Critic: Dueling architecture [1024, 512, 256, 128] + residual")
    
    # Start training
    print("\n[Step 3] Starting training...")
    print(f"Total Episodes: {args.n_episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Evaluation frequency: every {args.eval_freq} episodes")
    
    best_val_f1 = -np.inf
    best_val_accuracy = 0
    best_episode = 0
    
    for episode in range(args.n_episodes):
        state = train_env.reset()
        action = agent.select_action(state, deterministic=False)
        next_state, reward, done, info = train_env.step(action)
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        loss = agent.update(args.batch_size)
        if loss is not None:
            agent.train_losses.append(loss)
        
        agent.episode_rewards.append(reward)
        agent.episode_accuracies.append(1.0 if info['correct'] else 0.0)
        
        # Periodic evaluation
        val_metrics = None
        if (episode + 1) % args.eval_freq == 0:
            val_metrics = evaluate_agent(agent, val_env, n_episodes=len(y_val))
            
            print(f"\nEpisode {episode + 1}/{args.n_episodes}")
            print(f"  Lambda: {agent.lambda_temp.item() if torch.is_tensor(agent.lambda_temp) else agent.lambda_temp:.4f}")
            print(f"  Train Accuracy (last 1000): {np.mean(agent.episode_accuracies[-1000:]):.4f}")
            print(f"  Train Reward (last 1000): {np.mean(agent.episode_rewards[-1000:]):.4f}")
            print(f"  Validation Metrics:")
            print(f"    - Accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"    - Precision: {val_metrics['precision']:.4f}")
            print(f"    - Recall:    {val_metrics['recall']:.4f}")
            print(f"    - F1 Score:  {val_metrics['f1_score']:.4f}")
            print(f"    - Reward:    {val_metrics['avg_reward']:.4f}")
            
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_val_accuracy = val_metrics['accuracy']
                best_episode = episode + 1
                print(f"  New best F1 score: {best_val_f1:.4f} (Accuracy: {best_val_accuracy:.4f}, Episode {best_episode})")
                save_agent(agent, args.save_dir, 
                          episode=episode + 1, 
                          val_f1=val_metrics['f1_score'],
                          is_best=True)
    
    # Final evaluation
    print("\n[Step 4] Final evaluation...")
    final_val_metrics = evaluate_agent(agent, val_env, n_episodes=len(y_val))
    print(f"Final validation F1: {final_val_metrics['f1_score']:.4f}")
    print(f"Final validation accuracy: {final_val_metrics['accuracy']:.4f}")
    
    # Save final model
    print("\n[Step 5] Saving final model...")
    save_agent(agent, args.save_dir, 
              episode=args.n_episodes, 
              val_f1=final_val_metrics['f1_score'], 
              is_best=False)
    
    # Print training summary
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nTraining Summary:")
    print(f"  - Total Episodes: {args.n_episodes}")
    print(f"  - Best model (Episode {best_episode}):")
    print(f"      * F1 Score:  {best_val_f1:.4f}")
    print(f"      * Accuracy:  {best_val_accuracy:.4f}")
    print(f"  - Final model:")
    print(f"      * F1 Score:  {final_val_metrics['f1_score']:.4f}")
    print(f"      * Accuracy:  {final_val_metrics['accuracy']:.4f}")
    print(f"      * Precision: {final_val_metrics['precision']:.4f}")
    print(f"      * Recall:    {final_val_metrics['recall']:.4f}")
    print(f"\nModel files:")
    print(f"  - Best model: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"  - Final model: {os.path.join(args.save_dir, 'sac_model.pth')}")
    print("="*60)

if __name__ == '__main__':
    main()

