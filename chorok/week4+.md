# SALT 프로젝트 완전 실행 계획 (Week 4-12)

## 목표
RelBench SALT 데이터셋으로 **논문 제출 가능한 수준**의 연구 완성

---

## Week 4-5: SALT 데이터 탐색 및 이해

### Day 1-3: 데이터 다운로드 및 기초 분석

```python
# File: notebooks/01_salt_exploration.ipynb

"""
목표: SALT 데이터셋의 모든 특성 파악
"""

from relbench.datasets import get_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
dataset = get_dataset("rel-salt")

# 2. 기본 통계
print("=" * 60)
print("SALT Dataset Overview")
print("=" * 60)
print(f"Tables: {list(dataset.tables.keys())}")
print(f"Tasks: {len(dataset.tasks)}")

for table_name, table in dataset.tables.items():
    print(f"\n{table_name}:")
    print(f"  Rows: {len(table)}")
    print(f"  Columns: {list(table.columns)}")
    print(f"  Time range: {table.index.min()} ~ {table.index.max()}")

# 3. 각 Task별 분석
for task in dataset.tasks:
    print(f"\n{'='*60}")
    print(f"Task: {task.name}")
    print(f"{'='*60}")
    print(f"  Entity: {task.entity_table}")
    print(f"  Target: {task.target_col}")
    print(f"  Num classes: {task.num_classes}")
    
    # Class distribution
    train_table = dataset.get_task_table(task, split='train')
    class_dist = train_table[task.target_col].value_counts()
    
    print(f"\n  Class distribution:")
    for cls, count in class_dist.items():
        print(f"    {cls}: {count} ({count/len(train_table)*100:.1f}%)")
    
    # Check imbalance
    max_ratio = class_dist.max() / class_dist.min()
    print(f"  Imbalance ratio: {max_ratio:.2f}")
    if max_ratio > 10:
        print("  ⚠️  WARNING: Severe class imbalance!")

# 4. Temporal distribution analysis
def plot_temporal_distribution(dataset, task):
    """시간에 따른 데이터 분포 변화"""
    
    train_table = dataset.get_task_table(task, split='train')
    val_table = dataset.get_task_table(task, split='val')
    test_table = dataset.get_task_table(task, split='test')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (split_name, split_table) in zip(axes, [
        ('Train', train_table),
        ('Val', val_table),
        ('Test', test_table)
    ]):
        split_table[task.target_col].value_counts().plot(
            kind='bar', ax=ax, title=f'{split_name} Set'
        )
        ax.set_ylabel('Count')
        ax.set_xlabel('Class')
    
    plt.suptitle(f'Task: {task.name} - Temporal Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/temporal_dist_{task.name}.png', dpi=150)
    plt.show()

# Plot for all tasks
for task in dataset.tasks:
    plot_temporal_distribution(dataset, task)

# 5. Critical check: Distribution shift 감지
def check_distribution_shift(dataset, task):
    """
    Train vs Val vs Test의 분포 차이 측정
    Chi-square test 사용
    """
    from scipy.stats import chisquare
    
    train = dataset.get_task_table(task, split='train')[task.target_col].value_counts()
    val = dataset.get_task_table(task, split='val')[task.target_col].value_counts()
    test = dataset.get_task_table(task, split='test')[task.target_col].value_counts()
    
    # Align indices
    all_classes = sorted(set(train.index) | set(val.index) | set(test.index))
    train_aligned = [train.get(c, 0) for c in all_classes]
    val_aligned = [val.get(c, 0) for c in all_classes]
    test_aligned = [test.get(c, 0) for c in all_classes]
    
    # Chi-square tests
    _, p_train_val = chisquare(train_aligned, val_aligned)
    _, p_train_test = chisquare(train_aligned, test_aligned)
    _, p_val_test = chisquare(val_aligned, test_aligned)
    
    print(f"\n{task.name} Distribution Shift Test:")
    print(f"  Train vs Val: p={p_train_val:.4f} {'✅ Similar' if p_train_val > 0.05 else '⚠️  SHIFT!'}")
    print(f"  Train vs Test: p={p_train_test:.4f} {'✅ Similar' if p_train_test > 0.05 else '⚠️  SHIFT!'}")
    print(f"  Val vs Test: p={p_val_test:.4f} {'✅ Similar' if p_val_test > 0.05 else '⚠️  SHIFT!'}")
    
    return p_train_val, p_train_test, p_val_test

shift_results = {}
for task in dataset.tasks:
    shift_results[task.name] = check_distribution_shift(dataset, task)

# 6. Graph structure analysis
def analyze_graph_structure(dataset):
    """
    RelBench의 relational structure 분석
    """
    print("\n" + "="*60)
    print("Graph Structure Analysis")
    print("="*60)
    
    # Get foreign keys
    for table_name, table in dataset.tables.items():
        print(f"\n{table_name}:")
        # Check for foreign key columns (usually end with 'ID' or start with specific prefixes)
        fk_cols = [col for col in table.columns if 'PARTY' in col or col.endswith('ID')]
        print(f"  Potential foreign keys: {fk_cols}")
        
        # Check for NULL ratios
        for col in fk_cols:
            null_ratio = table[col].isna().mean()
            print(f"    {col}: {null_ratio*100:.1f}% NULL")

analyze_graph_structure(dataset)
```

**Week 4-5 체크포인트:**
- [ ] 8개 task 모두 이해
- [ ] Class imbalance 정도 파악
- [ ] Distribution shift 존재 여부 확인 (p-value)
- [ ] Graph structure (foreign keys) 파악
- [ ] **결정**: 이 데이터로 연구 가능한가? 심각한 문제 있나?

---

## Week 6-7: Baseline 모델 구현 및 검증

### Day 1-7: RelBench 기본 모델 실행

```python
# File: examples/salt_baseline.py

"""
목표: RelBench 기존 모델들의 성능 재현
- LightGBM (tabular)
- GNN (relational)
- 성능 차이 확인
"""

from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.nn import HeteroGraphSAGE
from relbench.modeling.loader import SparseTensor
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# ===== 1. LightGBM Baseline =====

class LightGBMBaseline:
    """
    Tabular baseline: 각 entity를 독립적으로 예측
    Graph structure 사용 안 함
    """
    def __init__(self, task):
        self.task = task
        self.model = None
    
    def prepare_features(self, dataset, split):
        """
        테이블을 flat feature로 변환
        """
        table = dataset.get_task_table(self.task, split=split)
        
        # Numeric features
        numeric_cols = table.select_dtypes(include=['float', 'int']).columns
        X = table[numeric_cols].fillna(0).values
        
        # Target
        y = table[self.task.target_col].values
        
        return X, y
    
    def train(self, dataset):
        """LightGBM 학습"""
        X_train, y_train = self.prepare_features(dataset, 'train')
        X_val, y_val = self.prepare_features(dataset, 'val')
        
        # LightGBM parameters
        params = {
            'objective': 'multiclass',
            'num_class': self.task.num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
    
    def evaluate(self, dataset, split='test'):
        """평가"""
        X, y = self.prepare_features(dataset, split)
        y_pred = self.model.predict(X).argmax(axis=1)
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        
        return {'accuracy': acc, 'f1': f1}


# ===== 2. GNN Baseline =====

class GNNBaseline:
    """
    Graph-based model: RelBench의 기본 GNN
    """
    def __init__(self, task, dataset):
        self.task = task
        self.dataset = dataset
        
        # Build graph
        self.data, self.col_stats_dict = make_pkey_fkey_graph(
            dataset.db,
            col_to_stype_dict=dataset.col_to_stype_dict,
            text_embedder_cfg=None,  # Skip text embedding for speed
        )
        
        # Model
        self.model = HeteroGraphSAGE(
            node_types=self.data.node_types,
            edge_types=self.data.edge_types,
            channels=128,
            aggr='mean',
            num_layers=2,
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, epochs=100):
        """GNN 학습"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Prepare data loader
        train_table = self.dataset.get_task_table(self.task, split='train')
        train_nodes = train_table.index.values
        
        train_loader = NeighborLoader(
            self.data,
            num_neighbors=[10, 5],
            batch_size=512,
            input_nodes=(self.task.entity_table, train_nodes),
            shuffle=True,
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward
                out = self.model(batch.x_dict, batch.edge_index_dict)
                target_out = out[self.task.entity_table]
                
                # Loss
                loss = criterion(target_out, batch[self.task.entity_table].y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                val_acc = self.evaluate(split='val')
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def evaluate(self, split='test'):
        """평가"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        test_table = self.dataset.get_task_table(self.task, split=split)
        test_nodes = test_table.index.values
        
        test_loader = NeighborLoader(
            self.data,
            num_neighbors=[10, 5],
            batch_size=512,
            input_nodes=(self.task.entity_table, test_nodes),
        )
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = self.model(batch.x_dict, batch.edge_index_dict)
                target_out = out[self.task.entity_table]
                
                preds = target_out.argmax(dim=1).cpu()
                labels = batch[self.task.entity_table].y.cpu()
                
                all_preds.append(preds)
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        acc = (all_preds == all_labels).float().mean().item()
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {'accuracy': acc, 'f1': f1}


# ===== Main Experiment =====

def run_baseline_experiment():
    """
    모든 8개 task에 대해 baseline 실행
    """
    dataset = get_dataset("rel-salt")
    
    results = []
    
    for task in dataset.tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task.name}")
        print(f"{'='*80}")
        
        # LightGBM
        print("\n[1/2] Training LightGBM...")
        lgb_model = LightGBMBaseline(task)
        lgb_model.train(dataset)
        lgb_results = lgb_model.evaluate(dataset, split='test')
        
        # GNN
        print("\n[2/2] Training GNN...")
        gnn_model = GNNBaseline(task, dataset)
        gnn_model.train(epochs=100)
        gnn_results = gnn_model.evaluate(split='test')
        
        # Save results
        results.append({
            'task': task.name,
            'lgb_accuracy': lgb_results['accuracy'],
            'lgb_f1': lgb_results['f1'],
            'gnn_accuracy': gnn_results['accuracy'],
            'gnn_f1': gnn_results['f1'],
            'improvement': gnn_results['accuracy'] - lgb_results['accuracy']
        })
        
        print(f"\nResults:")
        print(f"  LightGBM: Acc={lgb_results['accuracy']:.4f}, F1={lgb_results['f1']:.4f}")
        print(f"  GNN: Acc={gnn_results['accuracy']:.4f}, F1={gnn_results['f1']:.4f}")
        print(f"  Improvement: {gnn_results['accuracy'] - lgb_results['accuracy']:.4f}")
    
    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"Average LightGBM Accuracy: {df['lgb_accuracy'].mean():.4f}")
    print(f"Average GNN Accuracy: {df['gnn_accuracy'].mean():.4f}")
    print(f"Average Improvement: {df['improvement'].mean():.4f}")
    print(f"GNN wins on {(df['improvement'] > 0).sum()}/8 tasks")
    
    # Critical check
    if df['gnn_accuracy'].mean() <= df['lgb_accuracy'].mean():
        print("\n⚠️  WARNING: GNN is not better than LightGBM on average!")
        print("Consider:")
        print("  1. Graph structure may not be useful for these tasks")
        print("  2. Hyperparameter tuning needed")
        print("  3. Rethink research direction")
    
    df.to_csv('results/baseline_comparison.csv', index=False)
    return df

if __name__ == "__main__":
    results = run_baseline_experiment()
```

**Week 6-7 체크포인트:**
- [ ] 8개 task 모두 baseline 성능 측정
- [ ] LightGBM vs GNN 비교 완료
- [ ] **Critical**: GNN이 LightGBM보다 나은가?
- [ ] 결과를 표와 그래프로 정리
- [ ] **Go/No-Go 결정**: 연구 계속 진행할 가치 있는가?

---

## Week 8-10: Uncertainty Quantification 적용

### Week 8: MC Dropout + Ensembles

```python
# File: relbench/modeling/uncertainty.py

"""
SALT용 UQ 모듈
Week 2-3에서 배운 것을 SALT에 적용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class UncertaintyWrapper:
    """
    기존 모델을 UQ 가능하게 만드는 wrapper
    """
    def __init__(self, base_model, method='mc_dropout'):
        """
        Args:
            base_model: 학습된 GNN 모델
            method: 'mc_dropout', 'ensemble', or 'both'
        """
        self.base_model = base_model
        self.method = method
        self.enable_dropout_at_test = True
    
    def predict_with_uncertainty(
        self, 
        data, 
        n_samples=50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        불확실성과 함께 예측
        
        Returns:
            mean_probs: (N, C) - 평균 예측 확률
            epistemic: (N,) - Epistemic uncertainty
            aleatoric: (N,) - Aleatoric uncertainty
        """
        if self.method == 'mc_dropout':
            return self._mc_dropout_predict(data, n_samples)
        elif self.method == 'ensemble':
            return self._ensemble_predict(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mc_dropout_predict(self, data, n_samples):
        """MC Dropout 예측"""
        self.base_model.eval()
        
        all_probs = []
        for _ in range(n_samples):
            # Enable dropout at test time
            logits = self._forward_with_dropout(data)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        all_probs = torch.stack(all_probs)  # (n_samples, N, C)
        
        # Mean prediction
        mean_probs = all_probs.mean(dim=0)
        
        # Epistemic: variance across samples
        epistemic = all_probs.var(dim=0).mean(dim=1)
        
        # Aleatoric: entropy of mean prediction
        aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
        
        return mean_probs, epistemic, aleatoric
    
    def _forward_with_dropout(self, data):
        """
        Dropout을 활성화한 상태로 forward pass
        """
        # Find all dropout layers and enable them
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Enable dropout
        
        with torch.no_grad():
            logits = self.base_model(data.x_dict, data.edge_index_dict)
        
        return logits
    
    def calibrate(self, val_data, val_labels):
        """
        Temperature scaling으로 calibration
        """
        from week2_plan import TemperatureScaling
        
        # Get validation logits
        self.base_model.eval()
        with torch.no_grad():
            val_logits = self.base_model(val_data.x_dict, val_data.edge_index_dict)
        
        # Find optimal temperature
        temp_scaler = TemperatureScaling()
        optimal_temp = temp_scaler.calibrate_simple(val_logits, val_labels)
        
        self.temperature = optimal_temp
        print(f"Optimal temperature: {optimal_temp:.4f}")
        
        return optimal_temp


class MultiTaskUncertaintyModel(nn.Module):
    """
    Multi-task GNN with uncertainty correlation
    SALT의 8개 task를 동시에 예측
    """
    def __init__(self, base_encoder, task_configs):
        """
        Args:
            base_encoder: Shared GNN encoder
            task_configs: List of (task_name, num_classes) tuples
        """
        super().__init__()
        
        self.encoder = base_encoder
        self.num_tasks = len(task_configs)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs:
            self.task_heads[task_name] = nn.Linear(
                base_encoder.out_channels, 
                num_classes
            )
        
        # Uncertainty correlation (learnable)
        self.register_parameter(
            'uncertainty_correlation',
            nn.Parameter(torch.eye(self.num_tasks) * 0.5)
        )
    
    def forward(self, x_dict, edge_index_dict, task_names):
        """
        Multi-task forward pass
        
        Returns:
            predictions: Dict[task_name -> logits]
            uncertainties: Dict[task_name -> uncertainty]
        """
        # Shared encoding
        h = self.encoder(x_dict, edge_index_dict)
        
        predictions = {}
        raw_uncertainties = []
        
        for task_name in task_names:
            # Task-specific prediction
            logits = self.task_heads[task_name](h)
            predictions[task_name] = logits
            
            # Predictive uncertainty (entropy)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            raw_uncertainties.append(entropy)
        
        # Apply correlation
        raw_uncertainties = torch.stack(raw_uncertainties, dim=1)  # (N, num_tasks)
        
        # Correlation adjustment
        corr = torch.sigmoid(self.uncertainty_correlation)
        corr = (corr + corr.t()) / 2  # Symmetric
        
        adjusted_uncertainties = torch.matmul(raw_uncertainties, corr)
        
        uncertainties = {
            task_name: adjusted_uncertainties[:, i]
            for i, task_name in enumerate(task_names)
        }
        
        return predictions, uncertainties


# ===== Temporal Uncertainty Analysis for SALT =====

class SALTTemporalAnalyzer:
    """
    SALT 데이터의 temporal uncertainty 분석
    2018-2019 (stable) vs 2020 (COVID) 비교
    """
    def __init__(self, model, dataset, task):
        self.model = model
        self.dataset = dataset
        self.task = task
    
    def analyze_all_periods(self):
        """
        Train, Val, Test 각 기간의 uncertainty 분석
        """
        results = {}
        
        for split in ['train', 'val', 'test']:
            print(f"\nAnalyzing {split} split...")
            
            # Get data for this period
            split_data = self.dataset.get_task_table(self.task, split=split)
            
            # Predict with uncertainty
            mean_probs, epistemic, aleatoric = self.model.predict_with_uncertainty(
                split_data, n_samples=50
            )
            
            # Accuracy
            preds = mean_probs.argmax(dim=1)
            labels = split_data[self.task.target_col].values
            acc = (preds.cpu().numpy() == labels).mean()
            
            results[split] = {
                'accuracy': acc,
                'avg_epistemic': epistemic.mean().item(),
                'avg_aleatoric': aleatoric.mean().item(),
                'avg_total': (epistemic + aleatoric).mean().item(),
                'std_epistemic': epistemic.std().item(),
                'std_aleatoric': aleatoric.std().item(),
            }
            
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Avg Epistemic: {epistemic.mean():.4f}")
            print(f"  Avg Aleatoric: {aleatoric.mean():.4f}")
        
        # Check for distribution shift
        epistemic_increase = (
            results['test']['avg_epistemic'] - results['train']['avg_epistemic']
        ) / results['train']['avg_epistemic']
        
        print(f"\n{'='*60}")
        print("TEMPORAL SHIFT ANALYSIS")
        print(f"{'='*60}")
        print(f"Epistemic increase (train -> test): {epistemic_increase*100:.1f}%")
        
        if epistemic_increase > 0.2:
            print("✅ Significant epistemic increase detected!")
            print("   → Distribution shift likely (COVID-19 impact)")
        else:
            print("⚠️  No significant epistemic increase")
            print("   → Distribution may be stable")
        
        return results
    
    def plot_temporal_evolution(self, results):
        """
        시간에 따른 uncertainty 변화 시각화
        """
        import matplotlib.pyplot as plt
        
        splits = ['train', 'val', 'test']
        epistemic = [results[s]['avg_epistemic'] for s in splits]
        aleatoric = [results[s]['avg_aleatoric'] for s in splits]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(splits))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], epistemic, width, 
               label='Epistemic', alpha=0.8)
        ax.bar([i + width/2 for i in x], aleatoric, width,
               label='Aleatoric', alpha=0.8)
        
        ax.set_xlabel('Time Period', fontsize=14)
        ax.set_ylabel('Uncertainty', fontsize=14)
        ax.set_title(f'Temporal Uncertainty Evolution: {self.task.name}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(['2018-2020.02\n(Train)', 
                            '2020.02-07\n(Val)', 
                            '2020.07+\n(Test)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight COVID period
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(0.5, ax.get_ylim()[1]*0.9, 'COVID-19 ↓', 
                ha='center', fontsize=12, color='red')
        
        plt.tight_layout()
        plt.savefig(f'figures/temporal_uncertainty_{self.task.name}.png', dpi=150)
        plt.show()


# ===== Main Experiment =====

def run_uq_experiment():
    """
    Week 8-10: UQ 실험 전체 파이프라인
    """
    dataset = get_dataset("rel-salt")
    
    all_task_results = {}
    
    for task in dataset.tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task.name}")
        print(f"{'='*80}")
        
        # 1. Train base GNN
        print("\n[1/3] Training base GNN...")
        gnn_model = GNNBaseline(task, dataset)
        gnn_model.train(epochs=100)
        
        # 2. Wrap with uncertainty
        print("\n[2/3] Adding uncertainty quantification...")
        uq_model = UncertaintyWrapper(gnn_model.model, method='mc_dropout')
        
        # 3. Temporal analysis
        print("\n[3/3] Temporal uncertainty analysis...")
        analyzer = SALTTemporalAnalyzer(uq_model, dataset, task)
        results = analyzer.analyze_all_periods()
        analyzer.plot_temporal_evolution(results)
        
        all_task_results[task.name] = results
    
    # Summary across all tasks
    print("\n" + "="*80)
    print("SUMMARY: Epistemic Uncertainty Across All Tasks")
    print("="*80)
    
    for task_name, results in all_task_results.items():
        epistemic_shift = (
            results['test']['avg_epistemic'] - results['train']['avg_epistemic']
        )
        print(f"{task_name:30s}: {epistemic_shift:+.4f}")
    
    return all_task_results

if __name__ == "__main__":
    results = run_uq_experiment()
```

**Week 8-10 체크포인트:**
- [ ] 8개 task 모두 UQ 적용 완료
- [ ] Temporal analysis 완료 (train/val/test 비교)
- [ ] Epistemic increase가 유의미한가?
- [ ] Uncertainty vs Error correlation 확인
- [ ] Calibration (ECE) 측정
- [ ] 모든 결과를 표와 그래프로 정리

---

## Week 11-12: 논문 작성 및 마무리

### 논문 구조 (8-10 pages)

```markdown
# Uncertainty Quantification in Heterogeneous Graph Neural Networks for Enterprise Resource Planning

## Abstract (200 words)
- Problem: ERP prediction needs uncertainty for risk management
- Challenge: Relational data + temporal distribution shift
- Method: Uncertainty propagation in heterogeneous GNNs
- Result: X% better calibration, Y% shift detection
- Impact: Practical deployment guidelines

## 1. Introduction (1.5 pages)
### 1.1 Motivation
- ERP systems critical for business
- Autocomplete features need confidence estimates
- COVID-19 as natural distribution shift

### 1.2 Challenges
- Heterogeneous relational structure
- Multiple correlated tasks
- Temporal uncertainty evolution

### 1.3 Contributions
1. First UQ framework for heterogeneous GNNs on ERP data
2. Temporal uncertainty decomposition under distribution shift
3. Multi-task uncertainty correlation analysis
4. Comprehensive benchmark on SALT dataset (8 tasks)

## 2. Related Work (1.5 pages)
### 2.1 Uncertainty Quantification
- MC Dropout, Deep Ensembles
- Calibration methods

### 2.2 Graph Neural Networks
- GCN, GraphSAGE, heterogeneous GNNs
- Few works on GNN + UQ

### 2.3 ERP Prediction
- Existing ML approaches
- Gap: No UQ for relational ERP data

## 3. Method (2.5 pages)
### 3.1 Problem Formulation
- Heterogeneous graph definition
- Multi-task prediction setup

### 3.2 Base Model: HeteroGraphSAGE
- Architecture overview

### 3.3 Uncertainty Quantification
- MC Dropout for epistemic
- Predictive entropy for aleatoric
- Temperature scaling for calibration

### 3.4 Temporal Uncertainty Decomposition
- How uncertainty changes over time
- Distribution shift detection

### 3.5 Multi-Task Uncertainty Correlation
- Shared uncertainty patterns
- Correlation matrix learning

## 4. Experiments (2.5 pages)
### 4.1 Experimental Setup
- SALT dataset (8 tasks)
- Train/Val/Test splits (temporal)
- Evaluation metrics: Acc, ECE, NLL, Brier

### 4.2 Main Results
- Table: All tasks, all metrics
- UQ improves calibration by X%

### 4.3 RQ1: Temporal Uncertainty Evolution
- Figure: Epistemic increase during COVID
- Epistemic grows Y% from train to test

### 4.4 RQ2: Uncertainty vs Error
- Figure: High uncertainty → high error rate
- Correlation = 0.X

### 4.5 RQ3: Multi-Task Correlation
- Figure: Correlation heatmap
- Office ↔ Group strongly correlated

### 4.6 Ablation Study
- Table: Base vs +Dropout vs +Calibration
- Each component contributes

## 5. Analysis (1 page)
### 5.1 Failure Case Analysis
- When does the model have high uncertainty?
- Feature analysis

### 5.2 Business Impact Simulation
- Defer top 20% uncertain predictions → 80% error reduction
- Cost-benefit analysis

### 5.3 Computational Cost
- Overhead of MC Dropout: X ms per sample

## 6. Conclusion & Future Work (0.5 page)
- First comprehensive UQ study on relational ERP
- Temporal analysis reveals COVID impact
- Future: Causal uncertainty, active learning

## References (1 page)
- 30-40 references

## Appendix (optional)
- Additional experimental details
- More visualizations
- Hyperparameters
```

### Week 11-12 실행 계획

**Day 1-3: 결과 정리**
- [ ] 모든 실험 결과를 표로 정리
- [ ] 핵심 figure 10개 생성
- [ ] Statistical significance test

**Day 4-7: 초안 작성**
- [ ] Section 1-3 작성 (intro, related, method)
- [ ] Section 4 작성 (experiments)
- [ ] Abstract 작성

**Day 8-10: 수정 및 다듬기**
- [ ] 전체 읽고 논리 흐름 확인
- [ ] Figure caption 작성
- [ ] Related work 보강

**Day 11-14: 최종 마무리**
- [ ] 동료/지도교수 피드백 받기
- [ ] References 정리
- [ ] 코드 정리 및 GitHub 업로드
- [ ] Supplementary material 작성

---

## 최종 Deliverables

### 논문
- [ ] Main paper (8-10 pages)
- [ ] Supplementary material
- [ ] arXiv preprint

### 코드
- [ ] GitHub repository
  - README with instructions
  - requirements.txt
  - All experiment scripts
  - Pretrained models
  - Result reproduction guide

### 데이터
- [ ] Processed SALT data (if allowed)
- [ ] Result CSVs
- [ ] All figures (high-res)

---

## 성공 지표

### Minimum Viable Paper (Workshop)
- [ ] 8 tasks 모두 실험 완료
- [ ] Baseline 대비 ECE 개선
- [ ] Temporal shift 정량화
- [ ] 코드 공개

### Strong Paper (Main Conference)
- [ ] 위 + Ablation study
- [ ] 위 + Multi-task correlation
- [ ] 위 + Business impact analysis
- [ ] 위 + Theoretical justification

### Excellent Paper (Top Venue)
- [ ] 위 + Novel method (uncertainty propagation)
- [ ] 위 + 다른 데이터셋 검증
- [ ] 위 + Causal analysis
- [ ] 위 + 100+ citations in 2 years

---
## Risk Mitigation

### If GNN ≤ LightGBM:
→ Pivot to "When does graph structure help UQ?"

### If No temporal shift:
→ Focus on multi-task correlation

### If Everything fails:
→ Benchmark paper: "Comprehensive UQ evaluation on SALT"

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-3 | Learn GNN + UQ | Code mastery |
| 4-5 | SALT exploration | Data understanding |
| 6-7 | Baselines | Performance benchmark |
| 8-10 | UQ experiments | Core results |
| 11-12 | Paper writing | Submission-ready paper |

**Total: 12 weeks (3 months) full-time**

---

## Next Immediate Actions

1. **지금 당장**: Week 1 시작 (GNN 논문 읽기)
   - [ ] GCN 논문 다운로드 및 읽기 시작
   - [ ] PyTorch Geometric 설치
   - [ ] Cora 데이터셋 다운로드

2. **이번 주 안에**: Week 1 완료
   - [ ] 3개 GNN 논문 모두 읽고 요약
   - [ ] Cora에서 85% accuracy 달성
   - [ ] 결과 정리 및 다음 주 계획

3. **다음 주**: Week 2 시작 (UQ)
   - [ ] MC Dropout 논문 읽기
   - [ ] MNIST에서 UQ 구현

4. **한 달 후**: SALT 탐색 시작
   - [ ] RelBench 설치
   - [ ] SALT 데이터 다운로드
   - [ ] 첫 번째 baseline 실행

---

## 학습 자료 다운로드 링크

### Week 1: GNN 논문
1. **GCN**: https://arxiv.org/abs/1609.02907
2. **GraphSAGE**: https://arxiv.org/abs/1706.02216
3. **GAT**: https://arxiv.org/abs/1710.10903

### Week 2: UQ 논문
1. **MC Dropout**: https://arxiv.org/abs/1506.02142
2. **Deep Ensembles**: https://arxiv.org/abs/1612.01474
3. **Temperature Scaling**: https://arxiv.org/abs/1706.04599
4. **Conformal Prediction**: https://arxiv.org/abs/2107.07511

### Week 3: GNN + UQ
1. **Bayesian GCN**: https://arxiv.org/abs/1805.07140
2. **Graph Posterior Network**: https://arxiv.org/abs/2110.14012
3. **Evidential GNN**: (ICML Workshop 2021)

### 코드 저장소
1. **PyTorch Geometric**: https://github.com/pyg-team/pytorch_geometric
2. **RelBench**: https://github.com/snap-stanford/relbench
3. **Uncertainty Baselines**: https://github.com/google/uncertainty-baselines

### 온라인 강의
1. **CS224W (Stanford)**: http://web.stanford.edu/class/cs224w/
2. **Graph ML (McGill)**: https://cs.mcgill.ca/~wlh/comp766/
3. **Probabilistic ML**: https://www.cs.toronto.edu/~rgrosse/courses/csc2547_2021/

---

## Daily Progress Tracking Template

```markdown
# Day X Progress Log

## Date: YYYY-MM-DD
## Week: X | Phase: [Learning/Exploration/Experimentation/Writing]

### Today's Goals
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

### What I Did
- Spent X hours on Y
- Read paper Z (key insights: ...)
- Implemented feature W
- Results: ...

### Key Learnings
1. Important insight 1
2. Important insight 2

### Challenges
- Problem encountered: ...
- Solution attempt: ...
- Status: [Solved/Ongoing/Blocked]

### Tomorrow's Plan
- [ ] Task 1
- [ ] Task 2

### Questions/Notes
- Question for mentor: ...
- Interesting idea: ...

### Time Spent
- Reading: X hours
- Coding: Y hours
- Debugging: Z hours
- Writing: W hours
Total: XX hours
```

---

## Weekly Review Template

```markdown
# Week X Review

## Objective for This Week
[What was the main goal?]

## Achievements
- [ ] Planned task 1 - Status
- [ ] Planned task 2 - Status
- [ ] Planned task 3 - Status

## Key Results
- Metric 1: Value
- Metric 2: Value
- Figure/Table created: X

## Learnings
### Technical
1. ...
2. ...

### Research Process
1. ...
2. ...

## Challenges & Solutions
| Challenge | Solution | Status |
|-----------|----------|--------|
| ... | ... | ... |

## Next Week Plan
### Main Goal
[One sentence objective]

### Specific Tasks
1. [ ] Task 1 (Est: X hours)
2. [ ] Task 2 (Est: Y hours)
3. [ ] Task 3 (Est: Z hours)

### Risk Factors
- Potential blocker 1
- Mitigation: ...

## Reflection
What went well: ...
What to improve: ...
```

---

## Milestone Checklist

### Milestone 1: Foundation (Week 1-3)
- [ ] GNN 논문 5편 이상 읽음
- [ ] UQ 논문 5편 이상 읽음
- [ ] PyTorch Geometric 능숙하게 사용
- [ ] Cora에서 GNN + UQ 구현 완료
- [ ] 핵심 개념을 남에게 설명 가능

**Exit Criteria**: "GNN + UQ 전문가"라고 자신있게 말할 수 있음

---

### Milestone 2: SALT Exploration (Week 4-5)
- [ ] RelBench 설치 및 작동
- [ ] SALT 데이터 완전 이해
- [ ] EDA 노트북 작성 (10+ plots)
- [ ] Distribution shift 정량화
- [ ] Graph structure 분석 완료

**Exit Criteria**: "SALT 데이터의 전문가"

---

### Milestone 3: Baseline (Week 6-7)
- [ ] LightGBM 8개 task 성능 측정
- [ ] GNN 8개 task 성능 측정
- [ ] 결과 비교 표 작성
- [ ] **Go/No-Go 결정**: 연구 가치 확인

**Exit Criteria**: "이 연구가 왜 중요한지" 명확히 설명 가능

**Critical Decision Point**: 
- If GNN >> LightGBM → Continue
- If GNN ≈ LightGBM → Pivot to "when graph helps"
- If GNN < LightGBM → Reconsider research direction

---

### Milestone 4: UQ Implementation (Week 8-10)
- [ ] MC Dropout 8개 task 적용
- [ ] Calibration 측정 (ECE)
- [ ] Temporal analysis 완료
- [ ] Multi-task correlation 분석
- [ ] 모든 figure 생성 (10+)

**Exit Criteria**: 논문의 실험 section 완성 가능

---

### Milestone 5: Paper Draft (Week 11)
- [ ] Introduction 초안
- [ ] Method section 완성
- [ ] Experiments section 완성
- [ ] Abstract 초안
- [ ] 전체 논리 흐름 확인

**Exit Criteria**: 동료가 읽고 피드백 가능한 수준

---

### Milestone 6: Submission (Week 12)
- [ ] 논문 최종 수정
- [ ] 코드 정리 및 GitHub 업로드
- [ ] README 작성
- [ ] arXiv 제출
- [ ] Workshop/Conference 제출

**Exit Criteria**: 제출 완료!

---

## Motivation & Support

### When Stuck
1. **Take a break**: 30분 산책
2. **Ask for help**: Stack Overflow, Reddit, Twitter
3. **Read related work**: 다른 사람은 어떻게 해결했나?
4. **Simplify**: 문제를 더 작게 나누기
5. **Sleep on it**: 다음 날 다시 보면 해결책이 보임

### When Discouraged
Remember:
- 첫 논문은 완벽하지 않아도 됨
- 모든 전문가도 처음에는 초보였음
- 실패는 배움의 기회
- 작은 진전도 진전임
- 당신은 이미 많이 알고 있음

### Celebrate Small Wins
- ✅ 논문 1편 읽음 → 축하!
- ✅ 코드가 드디어 작동함 → 축하!
- ✅ 첫 실험 완료 → 축하!
- ✅ Figure 하나 완성 → 축하!

### Community
- Twitter: #GraphML, #UncertaintyQuantification 팔로우
- Reddit: r/MachineLearning 참여
- Discord/Slack: ML 커뮤니티 가입
- Conference: Virtual conference 참석

---

## Success Stories (Inspiration)

많은 첫 논문들이 비슷한 여정을 거쳤습니다:

1. **시작**: 기존 방법을 새로운 데이터셋에 적용
2. **발견**: 예상치 못한 패턴 발견
3. **개선**: 간단한 아이디어로 개선
4. **검증**: 체계적 실험으로 증명
5. **발표**: Workshop → Conference → Citation

당신의 프로젝트도 이 경로를 따를 것입니다!

---

## Final Checklist Before Starting

### Environment Setup
- [ ] Python 3.8+ 설치
- [ ] PyTorch 2.0+ 설치
- [ ] PyTorch Geometric 설치
- [ ] Jupyter Notebook 설치
- [ ] GPU 접근 가능 (선택사항)

### Tools
- [ ] Git 설치 및 GitHub 계정
- [ ] Paper management tool (Zotero/Mendeley)
- [ ] Note-taking app (Notion/Obsidian)
- [ ] Time tracking (Optional)

### Mindset
- [ ] 3개월 commitment 가능
- [ ] 실패해도 괜찮다는 마음
- [ ] 배우는 것을 즐기는 태도
- [ ] 꾸준함 > 완벽함

### Support System
- [ ] 지도교수/멘토 확보
- [ ] 동료 연구자 네트워크
- [ ] 정기적 체크인 일정

---

