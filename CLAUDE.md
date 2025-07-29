# Graph Neural Networks for Regression - Assistant Guidelines

This document provides comprehensive guidance for implementing graph neural networks (GNNs) for regression tasks.

## Graph Theory Fundamentals

### Core Concepts
- **Graphs**: Mathematical structures G = (V, E) where V represents nodes/vertices and E represents edges
- **Node Features**: Attributes associated with each node (e.g., atom properties in molecular graphs)
- **Edge Features**: Attributes associated with each edge (e.g., bond types, distances)
- **Graph-level Features**: Global properties of the entire graph
- **Adjacency Matrix**: Mathematical representation of graph connectivity
- **Neighborhood**: Set of nodes directly connected to a given node

### Graph Types for Regression
- **Homogeneous Graphs**: Single node and edge type
- **Heterogeneous Graphs**: Multiple node/edge types
- **Directed vs Undirected**: Edge directionality considerations
- **Weighted Graphs**: Edges with associated weights/strengths

## GNN Architectures for Regression

### Recommended Architectures

#### Graph Convolutional Networks (GCN)
- **Strengths**: Efficient, well-established, good baseline
- **Implementation**: Use `GCNConv` layers with proper normalization

#### Graph Attention Networks (GAT)
- **Use Case**: When attention mechanisms are beneficial
- **Strengths**: Learns importance weights for neighbors
- **Implementation**: Use `GATConv` with multi-head attention

#### GraphSAGE (Sample and Aggregate)
- **Use Case**: Large graphs, inductive learning
- **Strengths**: Scalable, handles unseen nodes
- **Implementation**: Use `SAGEConv` with appropriate aggregation

#### Graph Isomorphism Networks (GIN)
- **Use Case**: When theoretical expressiveness is important
- **Strengths**: Theoretically powerful for graph classification
- **Implementation**: Use `GINConv` with MLPs

### Model Design Guidelines

1. **Layer Stacking**: Start with 2-4 GNN layers to avoid over-smoothing
2. **Activation Functions**: Use ReLU, LeakyReLU, or ELU between layers
3. **Normalization**: Apply BatchNorm or LayerNorm after each GNN layer
4. **Readout Functions**: Use global pooling (mean, max, sum) for graph-level predictions
5. **Output Layer**: Linear layer(s) for regression output

## Data Handling Best Practices

### Graph Data Preprocessing

#### Node Features
- **Normalization**: StandardScaler or MinMaxScaler for continuous features
- **Categorical Encoding**: One-hot encoding for categorical node attributes
- **Missing Values**: Use mean imputation or learned embeddings
- **Feature Selection**: Remove highly correlated or irrelevant features

#### Edge Features
- **Distance Features**: Normalize spatial distances (0-1 or z-score)
- **Categorical Edges**: One-hot encode edge types
- **Edge Weights**: Normalize to prevent dominance of high-weight edges

#### Graph-level Preprocessing
- **Graph Size Normalization**: Consider normalizing by number of nodes/edges
- **Structural Features**: Add graph statistics (diameter, clustering coefficient)
- **Data Augmentation**: Node/edge dropout, subgraph sampling

### Dataset Splitting Strategies
- **Random Split**: Simple but may leak information
- **Scaffold Split**: Group similar graphs together (recommended for molecules)
- **Time-based Split**: For temporal data
- **Stratified Split**: Maintain target distribution across splits

## Code Assistance Guidelines

### PyTorch Geometric Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import DataLoader

class GraphRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        x = self.output(x)
        return x
```

### DGL Implementation

```python
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, GlobalAttentionPooling

class DGLGraphRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        
        # Global pooling
        gate_nn = nn.Linear(hidden_dim, 1)
        self.pooling = GlobalAttentionPooling(gate_nn)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        
        # Graph-level representation
        g_repr = self.pooling(g, h)
        return self.output(g_repr)
```

### Training Loop Template

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.squeeze(), batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y)
            
            total_loss += loss.item()
            predictions.extend(out.squeeze().cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    return total_loss / len(loader), predictions, targets
```

## Experiment Planning and Optimization

### Hyperparameter Tuning Strategies

#### Key Hyperparameters
1. **Learning Rate**: Start with 1e-3, use learning rate scheduling
2. **Hidden Dimensions**: 64, 128, 256 (depends on dataset size)
3. **Number of Layers**: 2-5 layers (avoid over-smoothing)
4. **Dropout Rate**: 0.1-0.5 for regularization
5. **Batch Size**: 32-128 (depends on GPU memory)

#### Optimization Techniques
- **Adam Optimizer**: Good default choice with weight decay
- **Learning Rate Scheduling**: ReduceLROnPlateau or CosineAnnealingLR
- **Early Stopping**: Monitor validation loss with patience
- **Gradient Clipping**: Prevent exploding gradients (clip_grad_norm_)

### Regularization Methods
1. **Dropout**: Apply to node features and final layers
2. **Weight Decay**: L2 regularization on model parameters
3. **Batch Normalization**: Stabilize training and act as regularizer
4. **Data Augmentation**: Node/edge dropout during training

### Cross-Validation Strategy
- **K-Fold CV**: Use 5-10 folds for robust evaluation
- **Nested CV**: Outer loop for model selection, inner for hyperparameter tuning
- **Stratified CV**: Maintain target distribution across folds

## Evaluation Metrics for Graph Regression

### Primary Metrics

#### Mean Absolute Error (MAE)
```python
def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))
```

#### Root Mean Square Error (RMSE)
```python
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))
```

#### R-squared (Coefficient of Determination)
```python
def r2_score(predictions, targets):
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return 1 - (ss_res / ss_tot)
```

### Additional Metrics

#### Mean Absolute Percentage Error (MAPE)
```python
def mape(predictions, targets):
    return torch.mean(torch.abs((targets - predictions) / targets)) * 100
```

#### Pearson Correlation Coefficient
```python
def pearson_correlation(predictions, targets):
    vx = predictions - torch.mean(predictions)
    vy = targets - torch.mean(targets)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
```

### Evaluation Best Practices

1. **Multiple Metrics**: Report MAE, RMSE, and R² together
2. **Statistical Significance**: Use confidence intervals or t-tests
3. **Residual Analysis**: Plot predictions vs targets and residuals
4. **Cross-validation**: Report mean and standard deviation across folds
5. **Domain-specific Metrics**: Use task-relevant evaluation measures

### Visualization and Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(predictions, targets, title="Predictions vs Targets"):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.6)
    
    # Perfect prediction line
    min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()

def plot_residuals(predictions, targets):
    residuals = targets - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
```

## Troubleshooting Common Issues

### Over-smoothing
- **Problem**: Node representations become too similar after many layers
- **Solutions**: Use fewer layers (2-4), add residual connections, use jumping knowledge

### Memory Issues
- **Problem**: Large graphs don't fit in GPU memory
- **Solutions**: Use mini-batching, gradient checkpointing, or model parallelism

### Poor Convergence
- **Problem**: Model doesn't learn effectively
- **Solutions**: Check learning rate, add normalization, verify data preprocessing

### Overfitting
- **Problem**: Good training performance, poor validation performance
- **Solutions**: Increase dropout, add weight decay, reduce model complexity, get more data