# Week 3: GNN + UQ 결합 - 논문의 핵심 기여

"""
목표: Relational Uncertainty Propagation 구현
이것이 당신의 main contribution!

핵심 아이디어:
- GNN에서 feature만 전파하는 게 아니라
- Uncertainty도 그래프를 따라 전파됨
- 이웃의 불확실성이 내 예측의 불확실성에 영향
"""
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
```

# 핵심 Contribution: Uncertainty Propagation

```
class UncertaintyPropagationLayer(MessagePassing):
    """
    새로운 GNN layer: Feature + Uncertainty 동시 전파
    
    핵심 메커니즘:
    1. Feature message passing (기존 GNN)
    2. Uncertainty message passing (새로운!)
    3. Uncertainty-weighted aggregation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        
        # Feature transformation
        self.lin_feature = nn.Linear(in_channels, out_channels)
        
        # Uncertainty estimation head
        self.lin_uncertainty = nn.Linear(in_channels, 1)
        
        # Uncertainty aggregation parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Balance parameter
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
        
        Returns:
            h: Updated features (N, out_channels)
            u: Updated uncertainties (N, 1)
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute node degree for normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Transform features
        h = self.lin_feature(x)
        
        # Estimate local uncertainty (before propagation)
        u_local = torch.sigmoid(self.lin_uncertainty(x))
        
        # Propagate both features and uncertainties
        h_prop = self.propagate(edge_index, x=h, u=u_local, norm=norm)
        u_prop = self.propagate_uncertainty(edge_index, u=u_local, norm=norm)
        
        # Combine local and propagated uncertainty
        u_final = self.alpha * u_local + (1 - self.alpha) * u_prop
        
        return h_prop, u_final
    
    def message(self, x_j, u_j, norm):
        """
        Feature message with uncertainty weighting
        
        핵심: 불확실한 이웃의 메시지는 덜 신뢰
        """
        # Uncertainty-weighted message
        # u_j가 높으면 (불확실하면) weight가 낮아짐
        weight = (1.0 - u_j) * norm.view(-1, 1)
        return weight * x_j
    
    def propagate_uncertainty(self, edge_index, u, norm):
        """
        Uncertainty propagation: Max pooling
        
        왜 max? 이웃 중 가장 불확실한 것이 전파되어야 함
        """
        row, col = edge_index
        u_neighbors = u[row]  # Source node uncertainties
        
        # Normalize and aggregate with max
        u_weighted = u_neighbors * norm.view(-1, 1)
        
        # Max pooling over neighbors
        u_agg = torch.zeros(u.size(0), 1, device=u.device)
        u_agg.scatter_reduce_(0, col.view(-1, 1).expand_as(u_weighted),
                              u_weighted, reduce='amax', include_self=False)
        
        return u_agg


class UncertaintyGNN(nn.Module):
    """
    Complete GNN with Uncertainty Propagation
    
    2-layer GNN with uncertainty at each layer
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        
        self.uq_conv1 = UncertaintyPropagationLayer(in_channels, hidden_channels)
        self.uq_conv2 = UncertaintyPropagationLayer(hidden_channels, out_channels)
        self.dropout = dropout
        
        # MC Dropout for epistemic uncertainty
        self.dropout_enabled = True
    
    def forward(self, x, edge_index, return_uncertainty=True):
        """
        Forward pass with uncertainty tracking
        
        Returns:
            logits: Class predictions (N, num_classes)
            epistemic: Epistemic uncertainty (N,)
            aleatoric: Aleatoric uncertainty (N,)
        """
        # Layer 1
        h1, u1 = self.uq_conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training or self.dropout_enabled)
        
        # Layer 2
        logits, u2 = self.uq_conv2(h1, edge_index)
        
        if return_uncertainty:
            # Aleatoric: From uncertainty propagation
            aleatoric = u2.squeeze(-1)
            
            # Epistemic: From MC Dropout (if enabled)
            if self.dropout_enabled and not self.training:
                epistemic = self._compute_epistemic(x, edge_index, n_samples=20)
            else:
                epistemic = torch.zeros_like(aleatoric)
            
            return logits, epistemic, aleatoric
        else:
            return logits
    
    def _compute_epistemic(self, x, edge_index, n_samples=20):
        """
        Compute epistemic uncertainty via MC Dropout
        """
        all_logits = []
        original_mode = self.training
        self.eval()
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits, _, _ = self.forward(x, edge_index, return_uncertainty=True)
                all_logits.append(F.softmax(logits, dim=1))
        
        self.train(original_mode)
        
        # Variance across samples
        all_logits = torch.stack(all_logits)  # (n_samples, N, C)
        epistemic = all_logits.var(dim=0).mean(dim=1)
        
        return epistemic
```


# Multi-Task Uncertainty Correlation

```
class MultiTaskUncertaintyGNN(nn.Module):
    """
    Multi-task GNN with uncertainty correlation
    
    핵심: 여러 task의 불확실성이 서로 관련됨
    예: office 예측이 불확실하면 group 예측도 불확실
    """
    def __init__(self, in_channels, hidden_channels, num_tasks, num_classes_per_task):
        super().__init__()
        
        # Shared encoder
        self.encoder = UncertaintyGNN(in_channels, hidden_channels, hidden_channels)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_channels, num_classes_per_task[i])
            for i in range(num_tasks)
        ])
        
        # Uncertainty correlation matrix (learnable)
        self.correlation_matrix = nn.Parameter(
            torch.eye(num_tasks) * 0.5 + torch.ones(num_tasks, num_tasks) * 0.1
        )
        
        self.num_tasks = num_tasks
    
    def forward(self, x, edge_index):
        """
        Returns:
            predictions: List of (N, C_i) for each task
            uncertainties: List of (N,) for each task
            corr_matrix: (num_tasks, num_tasks) correlation
        """
        # Shared encoding with uncertainty
        h, epistemic, aleatoric = self.encoder(x, edge_index)
        
        # Task-specific predictions
        predictions = []
        uncertainties = []
        
        for i, head in enumerate(self.task_heads):
            logits = head(h)
            predictions.append(logits)
            
            # Task-specific uncertainty (combine epistemic + aleatoric)
            task_uncertainty = epistemic + aleatoric
            uncertainties.append(task_uncertainty)
        
        # Apply correlation to uncertainties
        uncertainties_tensor = torch.stack(uncertainties, dim=1)  # (N, num_tasks)
        
        # Correlation-aware uncertainty adjustment
        corr_adjusted = self._apply_correlation(uncertainties_tensor)
        
        return predictions, corr_adjusted, self.correlation_matrix
    
    def _apply_correlation(self, uncertainties):
        """
        Apply learned correlation to uncertainties
        
        If task i and task j are correlated, high uncertainty in i
        should increase uncertainty in j
        """
        # Normalize correlation matrix (keep it symmetric and bounded)
        corr = torch.sigmoid(self.correlation_matrix)
        corr = (corr + corr.t()) / 2  # Enforce symmetry
        
        # Weighted sum: u_i' = sum_j(corr_ij * u_j)
        adjusted = torch.matmul(uncertainties, corr.t())
        
        return adjusted
    
    def get_uncertainty_correlation(self, data, mask):
        """
        Compute empirical uncertainty correlation on a dataset
        
        Returns:
            corr: (num_tasks, num_tasks) correlation matrix
        """
        self.eval()
        with torch.no_grad():
            _, uncertainties, _ = self.forward(data.x, data.edge_index)
            
            # Extract uncertainties for masked nodes
            u = uncertainties[mask].cpu().numpy()  # (N, num_tasks)
            
            # Compute correlation
            corr = np.corrcoef(u.T)  # (num_tasks, num_tasks)
        
        return corr

```

# Temporal Uncertainty Decomposition

```
class TemporalUncertaintyAnalyzer:
    """
    시간에 따른 불확실성 분해 및 분석
    
    핵심: Distribution shift 전후의 uncertainty 변화
    """
    def __init__(self, model):
        self.model = model
        self.history = {
            'epistemic': [],
            'aleatoric': [],
            'timestamps': []
        }
    
    def analyze_period(self, data, mask, timestamp):
        """
        특정 시간 구간의 불확실성 분석
        
        Args:
            data: Graph data
            mask: Boolean mask for this time period
            timestamp: Time identifier (e.g., '2019-Q1')
        """
        self.model.eval()
        with torch.no_grad():
            _, epistemic, aleatoric = self.model(data.x, data.edge_index)
            
            # Average over the period
            avg_epistemic = epistemic[mask].mean().item()
            avg_aleatoric = aleatoric[mask].mean().item()
            
            self.history['epistemic'].append(avg_epistemic)
            self.history['aleatoric'].append(avg_aleatoric)
            self.history['timestamps'].append(timestamp)
        
        return avg_epistemic, avg_aleatoric
    
    def detect_distribution_shift(self, threshold=0.1):
        """
        Distribution shift 감지
        
        Epistemic uncertainty가 급증하면 shift 의심
        """
        if len(self.history['epistemic']) < 2:
            return False, 0.0
        
        # Compare current vs previous
        current = self.history['epistemic'][-1]
        previous = self.history['epistemic'][-2]
        
        shift_magnitude = (current - previous) / (previous + 1e-10)
        
        is_shift = shift_magnitude > threshold
        
        return is_shift, shift_magnitude
    
    def plot_temporal_evolution(self):
        """
        시간에 따른 불확실성 변화 시각화
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        timestamps = self.history['timestamps']
        epistemic = self.history['epistemic']
        aleatoric = self.history['aleatoric']
        
        # Plot 1: Separate trends
        ax1.plot(timestamps, epistemic, 'o-', label='Epistemic', linewidth=2)
        ax1.plot(timestamps, aleatoric, 's-', label='Aleatoric', linewidth=2)
        ax1.set_xlabel('Time Period', fontsize=12)
        ax1.set_ylabel('Uncertainty', fontsize=12)
        ax1.set_title('Temporal Uncertainty Evolution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Stacked
        ax2.fill_between(range(len(timestamps)), 0, epistemic, 
                         alpha=0.5, label='Epistemic')
        ax2.fill_between(range(len(timestamps)), epistemic, 
                         [e+a for e,a in zip(epistemic, aleatoric)],
                         alpha=0.5, label='Aleatoric')
        ax2.set_xlabel('Time Period', fontsize=12)
        ax2.set_ylabel('Cumulative Uncertainty', fontsize=12)
        ax2.set_title('Uncertainty Decomposition', fontsize=14)
        ax2.set_xticks(range(len(timestamps)))
        ax2.set_xticklabels(timestamps, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_uncertainty.png', dpi=150)
        plt.show()
```


# Ablation Study Framework

```
class AblationStudy:
    """
    Systematic ablation study for uncertainty components
    
    비교할 variants:
    1. Base GNN (no uncertainty)
    2. + MC Dropout only
    3. + Uncertainty propagation only
    4. + Both (full model)
    """
    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes
        self.results = {}
    
    def run_variant(self, variant_name, model, epochs=200):
        """
        Train and evaluate one variant
        """
        print(f"\n{'='*60}")
        print(f"Running: {variant_name}")
        print(f"{'='*60}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        best_val_acc = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            if 'UncertaintyGNN' in model.__class__.__name__:
                logits, _, _ = model(self.data.x, self.data.edge_index)
            else:
                logits = model(self.data.x, self.data.edge_index)
            
            loss = criterion(logits[self.data.train_mask], 
                           self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            
            # Validation
            if (epoch + 1) % 50 == 0:
                val_acc = self.evaluate(model, self.data.val_mask)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Final evaluation
        test_metrics = self.evaluate_with_uncertainty(model)
        self.results[variant_name] = test_metrics
        
        return test_metrics
    
    def evaluate(self, model, mask):
        """Simple accuracy evaluation"""
        model.eval()
        with torch.no_grad():
            if 'UncertaintyGNN' in model.__class__.__name__:
                logits, _, _ = model(self.data.x, self.data.edge_index)
            else:
                logits = model(self.data.x, self.data.edge_index)
            
            pred = logits.argmax(dim=1)
            acc = (pred[mask] == self.data.y[mask]).float().mean()
        
        return acc.item()
    
    def evaluate_with_uncertainty(self, model):
        """Full evaluation with UQ metrics"""
        model.eval()
        
        with torch.no_grad():
            if 'UncertaintyGNN' in model.__class__.__name__:
                logits, epistemic, aleatoric = model(
                    self.data.x, self.data.edge_index
                )
            else:
                logits = model(self.data.x, self.data.edge_index)
                epistemic = torch.zeros(logits.size(0))
                aleatoric = torch.zeros(logits.size(0))
            
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            
            # Test set metrics
            test_mask = self.data.test_mask
            test_probs = probs[test_mask]
            test_labels = self.data.y[test_mask]
            
            # Accuracy
            acc = (pred[test_mask] == test_labels).float().mean().item()
            
            # UQ metrics
            from week2_plan import compute_ece, compute_nll, compute_brier_score
            
            ece = compute_ece(test_probs, test_labels)
            nll = compute_nll(test_probs, test_labels)
            brier = compute_brier_score(test_probs, test_labels, self.num_classes)
            
            # Uncertainty quality
            is_correct = (pred[test_mask] == test_labels).float()
            total_uncertainty = epistemic[test_mask] + aleatoric[test_mask]
            
            # Correlation between uncertainty and error
            if total_uncertainty.std() > 0:
                uncertainty_error_corr = torch.corrcoef(
                    torch.stack([total_uncertainty, 1 - is_correct])
                )[0, 1].item()
            else:
                uncertainty_error_corr = 0.0
        
        metrics = {
            'accuracy': acc,
            'ece': ece,
            'nll': nll,
            'brier': brier,
            'avg_epistemic': epistemic[test_mask].mean().item(),
            'avg_aleatoric': aleatoric[test_mask].mean().item(),
            'uncertainty_error_correlation': uncertainty_error_corr
        }
        
        return metrics
    
    def print_comparison(self):
        """
        Print comparison table of all variants
        """
        import pandas as pd
        
        df = pd.DataFrame(self.results).T
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        print(df.to_string())
        print("="*80)
        
        # Highlight best in each metric
        print("\nBest performers:")
        for metric in df.columns:
            if metric in ['ece', 'nll', 'brier']:  # Lower is better
                best = df[metric].idxmin()
            else:  # Higher is better
                best = df[metric].idxmax()
            print(f"  {metric}: {best} ({df.loc[best, metric]:.4f})")
        
        return df
```


# Week 3 Complete Example

```
def week3_complete_pipeline():
    """
    Week 3: GNN + UQ 결합 완전 파이프라인
    """
    from torch_geometric.datasets import Planetoid
    import numpy as np
    
    print("=" * 60)
    print("Week 3: GNN + Uncertainty Propagation")
    print("=" * 60)
    
    # Load data
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    
    # ===== 1. Train full UncertaintyGNN =====
    print("\n[1/3] Training UncertaintyGNN with propagation...")
    
    model = UncertaintyGNN(
        in_channels=dataset.num_node_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
        dropout=0.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        logits, epistemic, aleatoric = model(data.x, data.edge_index)
        
        # Main task loss
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        
        # Uncertainty regularization (optional)
        # Encourage low aleatoric where we have high confidence
        uncertainty_reg = (aleatoric[data.train_mask]).mean() * 0.01
        
        total_loss = loss + uncertainty_reg
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(data.x, data.edge_index)
                pred = logits.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # ===== 2. Uncertainty Analysis =====
    print("\n[2/3] Analyzing uncertainties...")
    
    model.eval()
    with torch.no_grad():
        logits, epistemic, aleatoric = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        
        test_mask = data.test_mask
        
        # Compute metrics
        is_correct = (pred[test_mask] == data.y[test_mask])
        
        print("\nUncertainty Statistics (Test Set):")
        print(f"  Avg Epistemic: {epistemic[test_mask].mean():.4f}")
        print(f"  Avg Aleatoric: {aleatoric[test_mask].mean():.4f}")
        print(f"  Avg Total: {(epistemic + aleatoric)[test_mask].mean():.4f}")
        
        # High vs Low uncertainty accuracy
        total_u = epistemic + aleatoric
        threshold = total_u[test_mask].median()
        
        high_u_mask = test_mask & (total_u > threshold)
        low_u_mask = test_mask & (total_u <= threshold)
        
        high_u_acc = (pred[high_u_mask] == data.y[high_u_mask]).float().mean()
        low_u_acc = (pred[low_u_mask] == data.y[low_u_mask]).float().mean()
        
        print(f"\nUncertainty vs Accuracy:")
        print(f"  High uncertainty nodes: {high_u_acc:.4f} accuracy")
        print(f"  Low uncertainty nodes: {low_u_acc:.4f} accuracy")
        print(f"  Difference: {(low_u_acc - high_u_acc):.4f}")
    
    # ===== 3. Ablation Study =====
    print("\n[3/3] Running ablation study...")
    
    from torch_geometric.nn import GCNConv
    
    class BaselineGNN(nn.Module):
        def __init__(self, in_c, hidden_c, out_c):
            super().__init__()
            self.conv1 = GCNConv(in_c, hidden_c)
            self.conv2 = GCNConv(hidden_c, out_c)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x
    
    ablation = AblationStudy(data, dataset.num_classes)
    
    # Variant 1: Baseline
    baseline = BaselineGNN(dataset.num_node_features, 16, dataset.num_classes)
    ablation.run_variant("Baseline GNN", baseline, epochs=200)
    
    # Variant 2: Full model (already trained)
    ablation.results["UncertaintyGNN (Full)"] = ablation.evaluate_with_uncertainty(model)
    
    # Print comparison
    df = ablation.print_comparison()
    
    return model, df
```


# Week 3 체크포인트


"""
✅ Week 3 완료 조건:

1. 핵심 기여 구현
   - [ ] UncertaintyPropagationLayer 구현 완료
   - [ ] Message passing에서 uncertainty weighting 작동 확인
   - [ ] Cora에서 uncertainty propagation 효과 검증

2. 실험
   - [ ] Ablation study: Baseline vs Full model
   - [ ] Uncertainty propagation 효과 정량화
   - [ ] High/Low uncertainty 샘플의 accuracy 차이 확인

3. 분석
   - [ ] Uncertainty가 그래프를 따라 전파되는지 시각화
   - [ ] 어떤 노드가 높은 uncertainty를 가지는지 패턴 발견
   - [ ] Epistemic vs Aleatoric 분해가 의미있는지 검증

4. 이해도
   - [ ] "Uncertainty propagation이 왜 필요한가?" 설명 가능
   - [ ] "Baseline GNN 대비 무엇이 나아졌나?" 답변 가능
   - [ ] 논문의 main contribution을 명확히 설명 가능

모두 체크되면 Week 4-8 (SALT 프로젝트)로!
"""
