# @title core architecture 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class GraphConfig:
    """Configuration for the Relational Graph Transformer"""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_hop_distance: int = 3
    num_centroids: int = 32
    local_attention_radius: int = 2
    use_temporal: bool = True
    use_schema_types: bool = True
    centroid_update_momentum: float = 0.99

class MultiElementTokenizer(nn.Module):
    """
    Converts graph nodes into multi-element tokens with 5 components:
    1. Features: Raw node attributes
    2. Type: Entity type from schema
    3. Hop Distance: Distance from reference node
    4. Time: Temporal information
    5. Local Structure: Local graph context
    """
    
    def __init__(self, config: GraphConfig, feature_dim: int, num_node_types: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.num_node_types = num_node_types
        
        # Feature embedding
        self.feature_proj = nn.Linear(feature_dim, config.hidden_dim // 4)
        
        # Type embedding
        self.type_embedding = nn.Embedding(num_node_types, config.hidden_dim // 4)
        
        # Hop distance embedding
        self.hop_embedding = nn.Embedding(config.max_hop_distance + 1, config.hidden_dim // 8)
        
        # Temporal embedding (if used)
        if config.use_temporal:
            self.temporal_proj = nn.Linear(1, config.hidden_dim // 8)
        
        # Local structure encoder - expects same dimension as feature projection output
        self.structure_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Final projection to hidden dimension
        # Components: features + types + structure + hop + (temporal if used)
        total_dim = config.hidden_dim // 4 * 3 + config.hidden_dim // 8  # Base components
        if config.use_temporal:
            total_dim += config.hidden_dim // 8  # Add temporal dimension
            
        self.final_proj = nn.Linear(total_dim, config.hidden_dim)
        
    def encode_local_structure(self, x, edge_index, batch_size):
        """Encode local graph structure using degree and neighbor features"""
        # Simple local structure: node degree and mean neighbor features
        row, col = edge_index
        deg = degree(row, num_nodes=x.size(0))
        
        # Aggregate neighbor features
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, row, x[col])
        
        # Avoid division by zero
        deg_safe = deg.clamp(min=1).unsqueeze(-1)
        neighbor_mean = neighbor_sum / deg_safe
        
        # Use only neighbor features for structure encoding to maintain dimensionality
        # The degree information is implicitly captured in the neighbor aggregation
        structure_features = neighbor_mean
        
        return self.structure_encoder(structure_features)
    
    def forward(self, x, node_types, hop_distances, edge_index, timestamps=None):
        """
        Args:
            x: Node features [num_nodes, feature_dim]
            node_types: Node type indices [num_nodes]
            hop_distances: Distance from reference nodes [num_nodes]
            edge_index: Graph edges [2, num_edges]
            timestamps: Temporal information [num_nodes] (optional)
        """
        batch_size = x.size(0)
        
        # 1. Feature embedding
        feature_emb = self.feature_proj(x)
        
        # 2. Type embedding
        type_emb = self.type_embedding(node_types)
        
        # 3. Hop distance embedding
        hop_emb = self.hop_embedding(hop_distances.clamp(0, self.config.max_hop_distance))
        
        # 4. Local structure embedding
        structure_emb = self.encode_local_structure(feature_emb, edge_index, batch_size)
        
        # Combine embeddings
        embeddings = [feature_emb, type_emb, structure_emb, hop_emb]
        
        # 5. Temporal embedding (if available)
        if self.config.use_temporal and timestamps is not None:
            temporal_emb = self.temporal_proj(timestamps.unsqueeze(-1))
            embeddings.append(temporal_emb)
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Final projection
        return self.final_proj(combined)

class LocalAttentionModule(nn.Module):
    """
    Local attention within neighborhoods defined by sampled subgraphs
    or schema-based relationships
    """
    
    def __init__(self, config: GraphConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def get_local_neighbors(self, edge_index, num_nodes, radius=2):
        """Get local neighborhoods within specified radius"""
        # Simple k-hop neighborhood extraction
        current_edges = edge_index
        all_neighbors = {}
        
        for node in range(num_nodes):
            neighbors = set([node])
            current_hop_nodes = set([node])
            
            for hop in range(radius):
                next_hop_nodes = set()
                for n in current_hop_nodes:
                    # Find neighbors of current hop nodes
                    mask = (current_edges[0] == n) | (current_edges[1] == n)
                    if mask.any():
                        edge_neighbors = torch.cat([
                            current_edges[1][current_edges[0] == n],
                            current_edges[0][current_edges[1] == n]
                        ]).unique()
                        next_hop_nodes.update(edge_neighbors.tolist())
                
                neighbors.update(next_hop_nodes)
                current_hop_nodes = next_hop_nodes
            
            all_neighbors[node] = list(neighbors)
        
        return all_neighbors
    
    def forward(self, x, edge_index, node_mask=None):
        """
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            node_mask: Optional mask for valid nodes
        """
        batch_size, hidden_dim = x.shape
        residual = x
        
        # Get local neighborhoods
        local_neighbors = self.get_local_neighbors(
            edge_index, batch_size, self.config.local_attention_radius
        )
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Apply local attention for each node
        attended_values = torch.zeros_like(V)
        
        for node_idx, neighbors in local_neighbors.items():
            if len(neighbors) == 0:
                # If no neighbors, use self-attention
                attended_values[node_idx] = V[node_idx]
                continue
                
            neighbors = torch.tensor(neighbors, device=x.device)
            
            # Perform attention computation per head
            node_attended = torch.zeros(self.num_heads, self.head_dim, device=x.device)
            
            for head in range(self.num_heads):
                # Get Q, K, V for this head
                q_node = Q[node_idx, head, :]  # [head_dim]
                k_neighbors = K[neighbors, head, :]  # [num_neighbors, head_dim]
                v_neighbors = V[neighbors, head, :]  # [num_neighbors, head_dim]
                
                # Compute attention scores
                scores = torch.matmul(q_node.unsqueeze(0), k_neighbors.transpose(0, 1))  # [1, num_neighbors]
                scores = scores / math.sqrt(self.head_dim)
                
                # Apply softmax
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Apply attention to values
                attended_head = torch.matmul(attn_weights, v_neighbors)  # [1, head_dim]
                node_attended[head] = attended_head.squeeze(0)
            
            attended_values[node_idx] = node_attended
        
        # Reshape and project output
        attended_values = attended_values.view(batch_size, hidden_dim)
        output = self.out_proj(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output

class GlobalAttentionModule(nn.Module):
    """
    Global attention to learnable centroids representing database-wide patterns
    """
    
    def __init__(self, config: GraphConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_centroids = config.num_centroids
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Learnable centroids
        self.centroids = nn.Parameter(torch.randn(config.num_centroids, config.hidden_dim))
        
        # Attention projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # For centroid updates (exponential moving average)
        self.momentum = config.centroid_update_momentum
        self.register_buffer('centroid_counts', torch.zeros(config.num_centroids))
        
    def update_centroids(self, x):
        """Update centroids using exponential moving average K-means"""
        with torch.no_grad():
            # Compute distances to all centroids
            distances = torch.cdist(x, self.centroids)  # [num_nodes, num_centroids]
            
            # Assign nodes to closest centroids
            assignments = distances.argmin(dim=1)  # [num_nodes]
            
            # Update centroids with exponential moving average
            for c in range(self.num_centroids):
                mask = (assignments == c)
                if mask.sum() > 0:
                    # Get nodes assigned to this centroid
                    assigned_nodes = x[mask]
                    new_centroid = assigned_nodes.mean(dim=0)
                    
                    # Exponential moving average update
                    self.centroids[c] = (self.momentum * self.centroids[c] + 
                                       (1 - self.momentum) * new_centroid)
                    
                    # Update counts
                    self.centroid_counts[c] = (self.momentum * self.centroid_counts[c] + 
                                             (1 - self.momentum) * mask.sum().float())
    
    def forward(self, x):
        """
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
        """
        batch_size = x.size(0)
        residual = x
        
        # Update centroids during training
        if self.training:
            self.update_centroids(x)
        
        # Compute Q from nodes, K and V from centroids
        Q = self.q_proj(x)  # [batch_size, hidden_dim]
        K = self.k_proj(self.centroids)  # [num_centroids, hidden_dim]
        V = self.v_proj(self.centroids)  # [num_centroids, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        K = K.view(self.num_centroids, self.num_heads, self.head_dim)  # [num_centroids, num_heads, head_dim]
        V = V.view(self.num_centroids, self.num_heads, self.head_dim)  # [num_centroids, num_heads, head_dim]
        
        # Compute attention scores for each head
        attended_values = []
        for h in range(self.num_heads):
            Q_h = Q[:, h, :]  # [batch_size, head_dim]
            K_h = K[:, h, :]  # [num_centroids, head_dim]
            V_h = V[:, h, :]  # [num_centroids, head_dim]
            
            # Attention scores: [batch_size, num_centroids]
            scores = torch.matmul(Q_h, K_h.transpose(0, 1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention: [batch_size, head_dim]
            attended = torch.matmul(attn_weights, V_h)
            attended_values.append(attended)
        
        # Concatenate heads
        global_context = torch.cat(attended_values, dim=-1)  # [batch_size, hidden_dim]
        
        # Project output
        output = self.out_proj(global_context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output

class FeedForwardNetwork(nn.Module):
    """Standard transformer feed-forward network"""
    
    def __init__(self, config: GraphConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.linear2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.layer_norm(x + residual)

class HybridTransformerLayer(nn.Module):
    """
    Single layer combining local attention, global attention, and FFN
    """
    
    def __init__(self, config: GraphConfig):
        super().__init__()
        self.config = config
        
        self.local_attention = LocalAttentionModule(config)
        self.global_attention = GlobalAttentionModule(config)
        self.ffn = FeedForwardNetwork(config)
        
        # Combination layer for local and global representations
        self.combination = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]
        """
        residual = x
        
        # Local attention within neighborhoods
        local_repr = self.local_attention(x, edge_index)
        
        # Global attention to centroids
        global_repr = self.global_attention(x)
        
        # Combine local and global representations
        combined = torch.cat([local_repr, global_repr], dim=-1)
        hybrid_repr = self.combination(combined)
        
        # Residual connection
        hybrid_repr = self.layer_norm(hybrid_repr + residual)
        
        # Feed-forward network
        output = self.ffn(hybrid_repr)
        
        return output

class RelationalGraphTransformer(nn.Module):
    """
    Complete Hybrid Local-Global Relational Graph Transformer
    """
    
    def __init__(self, config: GraphConfig, feature_dim: int, num_node_types: int, num_classes: int):
        super().__init__()
        self.config = config
        
        # Multi-element tokenizer
        self.tokenizer = MultiElementTokenizer(config, feature_dim, num_node_types)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            HybridTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, num_classes)
        )
        
        # Schema-aware edge type embeddings (optional)
        if config.use_schema_types:
            self.edge_type_embedding = nn.Embedding(10, config.hidden_dim // 8)  # Assume max 10 edge types
    
    def compute_hop_distances(self, edge_index, num_nodes, reference_nodes=None):
        """Compute hop distances from reference nodes"""
        if reference_nodes is None:
            # Use random reference nodes if not provided
            reference_nodes = torch.randint(0, num_nodes, (min(num_nodes // 10, 10),))
        
        # Simple BFS to compute distances (can be optimized)
        distances = torch.full((num_nodes,), self.config.max_hop_distance, dtype=torch.long)
        
        for ref_node in reference_nodes:
            visited = torch.zeros(num_nodes, dtype=torch.bool)
            queue = [(ref_node.item(), 0)]
            visited[ref_node] = True
            distances[ref_node] = 0
            
            while queue:
                node, dist = queue.pop(0)
                if dist >= self.config.max_hop_distance:
                    continue
                
                # Find neighbors
                neighbors = []
                mask = (edge_index[0] == node) | (edge_index[1] == node)
                if mask.any():
                    edge_neighbors = torch.cat([
                        edge_index[1][edge_index[0] == node],
                        edge_index[0][edge_index[1] == node]
                    ]).unique()
                    neighbors = edge_neighbors.tolist()
                
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        distances[neighbor] = min(distances[neighbor], dist + 1)
                        queue.append((neighbor, dist + 1))
        
        return distances.to(edge_index.device)
    
    def forward(self, x, edge_index, node_types, timestamps=None, reference_nodes=None):
        """
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Graph edges [2, num_edges]
            node_types: Node type indices [num_nodes]
            timestamps: Temporal information [num_nodes] (optional)
            reference_nodes: Reference nodes for hop distance computation (optional)
        """
        num_nodes = x.size(0)
        
        # Compute hop distances
        hop_distances = self.compute_hop_distances(edge_index, num_nodes, reference_nodes)
        
        # Multi-element tokenization
        x = self.tokenizer(x, node_types, hop_distances, edge_index, timestamps)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Output projection
        output = self.output_head(x)
        
        return output
    
    def get_node_embeddings(self, x, edge_index, node_types, timestamps=None, reference_nodes=None):
        """Get node embeddings without classification head"""
        num_nodes = x.size(0)
        
        # Compute hop distances
        hop_distances = self.compute_hop_distances(edge_index, num_nodes, reference_nodes)
        
        # Multi-element tokenization
        x = self.tokenizer(x, node_types, hop_distances, edge_index, timestamps)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        return x

# Example usage and training setup
def create_sample_graph():
    """Create a sample relational graph for testing"""
    num_nodes = 100
    feature_dim = 64
    num_node_types = 5
    num_classes = 3
    
    # Random node features
    x = torch.randn(num_nodes, feature_dim)
    
    # Random node types
    node_types = torch.randint(0, num_node_types, (num_nodes,))
    
    # Random graph structure
    edge_index = torch.randint(0, num_nodes, (2, 200))
    
    # Random timestamps
    timestamps = torch.randn(num_nodes)
    
    # Random labels
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    return x, edge_index, node_types, timestamps, labels

def train_model():
    """Example training loop"""
    # Configuration
    config = GraphConfig(
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        num_centroids=16
    )
    
    # Create sample data
    x, edge_index, node_types, timestamps, labels = create_sample_graph()
    
    # Initialize model
    model = RelationalGraphTransformer(
        config=config,
        feature_dim=x.size(-1),
        num_node_types=node_types.max().item() + 1,
        num_classes=labels.max().item() + 1
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x, edge_index, node_types, timestamps)
        
        # Compute loss
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    
    # Get node embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.get_node_embeddings(x, edge_index, node_types, timestamps)
        print(f"Node embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    train_model()
