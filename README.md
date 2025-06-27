# Relational Graph Transformer (RGT)

A PyTorch implementation of the **Relational Graph Transformer** architecture for learning on multi-table relational data represented as heterogeneous temporal graphs.

## 📖 Overview

This implementation is inspired by the paper ["Relational Graph Transformer"](https://arxiv.org/abs/2505.10960) by Dwivedi et al. Relational Deep Learning (RDL) is a promising approach for building state-of-the-art predictive models on multi-table relational data by representing it as a heterogeneous temporal graph. The paper addresses fundamental limitations of traditional Graph Neural Networks in capturing complex structural patterns and long-range dependencies inherent in relational databases.

## 🔑 Key Features

### 🎯 Multi-Element Tokenization
Our implementation features a novel **5-component tokenization scheme** that converts graph nodes into rich, multi-dimensional representations:

1. **Features**: Raw node attributes and embeddings
2. **Type**: Entity type information from database schema  
3. **Hop Distance**: Distance from reference nodes for positional encoding
4. **Time**: Temporal information for dynamic graphs
5. **Local Structure**: Local graph context through neighbor aggregation

### 🔄 Hybrid Attention Mechanism
The architecture combines two complementary attention mechanisms:

- **Local Attention**: Captures fine-grained relationships within k-hop neighborhoods
- **Global Attention**: Models database-wide patterns through learnable centroids with exponential moving average updates

### ⚡ Key Architectural Components

- **Configurable Architecture**: Flexible configuration system for easy experimentation
- **Schema-Aware Processing**: Leverages database schema information for better representations
- **Temporal Support**: Built-in support for temporal graph evolution
- **Scalable Design**: Efficient attention mechanisms for large-scale relational data

## 🚀 Installation

```bash
pip install torch torch-geometric numpy
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- NumPy

## 🛠️ Usage

### Basic Example

```python
import torch
from relational_graph_transformer import RelationalGraphTransformer, GraphConfig

# Configure the model
config = GraphConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    max_hop_distance=3,
    num_centroids=32,
    local_attention_radius=2,
    use_temporal=True,
    use_schema_types=True
)

# Initialize model
model = RelationalGraphTransformer(
    config=config,
    feature_dim=64,        # Input feature dimension
    num_node_types=5,      # Number of entity types in schema
    num_classes=3          # Number of output classes
)

# Sample data
x = torch.randn(100, 64)              # Node features
edge_index = torch.randint(0, 100, (2, 200))  # Graph edges
node_types = torch.randint(0, 5, (100,))      # Entity types
timestamps = torch.randn(100)                  # Temporal info

# Forward pass
output = model(x, edge_index, node_types, timestamps)
print(f"Output shape: {output.shape}")  # [100, 3]

# Get node embeddings (without classification head)
embeddings = model.get_node_embeddings(x, edge_index, node_types, timestamps)
print(f"Embedding shape: {embeddings.shape}")  # [100, 256]
```

### Configuration Options

```python
@dataclass
class GraphConfig:
    hidden_dim: int = 256                    # Hidden dimension size
    num_heads: int = 8                       # Number of attention heads
    num_layers: int = 6                      # Number of transformer layers
    dropout: float = 0.1                     # Dropout rate
    max_hop_distance: int = 3                # Maximum hop distance for positional encoding
    num_centroids: int = 32                  # Number of global attention centroids
    local_attention_radius: int = 2          # Local attention neighborhood radius
    use_temporal: bool = True                # Enable temporal features
    use_schema_types: bool = True            # Enable schema-aware processing
    centroid_update_momentum: float = 0.99   # EMA momentum for centroid updates
```

### Training Example

```python
# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

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
```

## 🏗️ Architecture Details

### Multi-Element Tokenizer
- **Feature Projection**: Linear transformation of raw node attributes
- **Type Embedding**: Learnable embeddings for entity types from database schema
- **Hop Distance Encoding**: Positional encoding based on graph distance
- **Temporal Encoding**: Time-aware representations for dynamic graphs
- **Structure Encoding**: Local neighborhood context through degree and neighbor aggregation

### Hybrid Transformer Layer
Each transformer layer consists of:
1. **Local Attention Module**: Attends to k-hop neighborhoods
2. **Global Attention Module**: Attends to learnable centroids
3. **Combination Layer**: Fuses local and global representations
4. **Feed-Forward Network**: Standard transformer FFN with residual connections

### Global Attention with Learnable Centroids
- Maintains learnable centroids representing database-wide patterns
- Updates centroids using exponential moving average during training
- Enables efficient modeling of long-range dependencies

## 📊 Applications

This implementation is suitable for various relational learning tasks:

- **Node Classification**: Predicting entity types or properties
- **Link Prediction**: Forecasting new relationships
- **Graph Classification**: Classifying entire relational graphs
- **Temporal Prediction**: Modeling evolution of relational data
- **Recommendation Systems**: Learning user-item interactions
- **Knowledge Graph Completion**: Predicting missing facts

## 🎯 Key Innovations

1. **Multi-Element Tokenization**: Novel 5-component token representation
2. **Hybrid Attention**: Combines local neighborhood and global pattern modeling
3. **Schema-Aware Processing**: Leverages database schema for better representations
4. **Temporal Support**: Built-in temporal modeling capabilities
5. **Scalable Design**: Efficient attention mechanisms for large graphs

## 📈 Performance Considerations

- **Memory Efficiency**: Global attention reduces quadratic complexity
- **Scalability**: Local attention limits neighborhood size
- **Flexibility**: Configurable architecture for different use cases
- **Generalization**: Schema-aware design improves transfer learning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{dwivedi2025relational,
  title={Relational Graph Transformer},
  author={Dwivedi, Vijay Prakash and Jaladi, Sri and Shen, Yangyi and L{\'o}pez, Federico and Kanatsoulis, Charilaos I. and Puri, Rishi and Fey, Matthias and Leskovec, Jure},
  journal={arXiv preprint arXiv:2505.10960},
  year={2025}
}
```

## 🔗 References

- [Original Paper: Relational Graph Transformer](https://arxiv.org/abs/2505.10960)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Graph Transformers Survey](https://arxiv.org/abs/2012.09699)

## 📞 Contact

For questions or support, please open an issue on GitHub or contact [sidhtangduggal511@gmail.com].

---

**Note**: This implementation is based on the research paper and may differ from the official implementation. It serves as a educational and research tool for understanding relational graph transformers.
