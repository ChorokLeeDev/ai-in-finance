# Week 2: Uncertainty Quantification 마스터

## 목표: UQ의 4가지 핵심 기법을 코드로 구현
- MC Dropout
- Deep Ensembles
- Temperature Scaling (Calibration)
- Conformal Prediction

## 📓 Runnable Code
**모든 코드는 실행 가능한 Jupyter Notebook으로 제공됩니다:**
- 파일: `week2_uq.ipynb`
- 각 섹션별로 실행하면서 학습하세요!

---

## 📅 Week 2 Daily Plan

### Day 1-2: MC Dropout
- [ ] Read MC Dropout paper (Gal & Ghahramani, 2016)
- [ ] Run the MC Dropout code on Cora
- [ ] Experiment with n_samples = [10, 50, 100]
- [ ] Understand epistemic uncertainty from dropout

### Day 3-4: Deep Ensembles
- [ ] Read Deep Ensembles paper (Lakshminarayanan et al., 2017)
- [ ] Run the ensemble code
- [ ] Experiment with n_models = [3, 5, 10]
- [ ] Compare with MC Dropout

### Day 5: Temperature Scaling
- [ ] Read Temperature Scaling paper (Guo et al., 2017)
- [ ] Apply calibration to your MC Dropout model
- [ ] Check if ECE improves
- [ ] Understand calibration importance

### Day 6-7: Conformal Prediction
- [ ] Read Conformal Prediction tutorial
- [ ] Run conformal prediction code
- [ ] Check coverage guarantees
- [ ] Understand distribution-free uncertainty

### Final: Compare All Methods
- [ ] Run the complete pipeline at the end of `week2_uq.ipynb`
- [ ] Compare accuracy, ECE, NLL, Brier Score
- [ ] Generate all visualizations
- [ ] Write summary of when to use each method

---

## 📚 Required Reading

1. **MC Dropout**: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
   - Paper: https://arxiv.org/abs/1506.02142

2. **Deep Ensembles**: Lakshminarayanan et al. (2017) - "Simple and Scalable Predictive Uncertainty Estimation"
   - Paper: https://arxiv.org/abs/1612.01474

3. **Temperature Scaling**: Guo et al. (2017) - "On Calibration of Modern Neural Networks"
   - Paper: https://arxiv.org/abs/1706.04599

4. **Conformal Prediction**: Angelopoulos & Bates (2021) - "A Gentle Introduction to Conformal Prediction"
   - Paper: https://arxiv.org/abs/2107.07511

---

## 💡 Key Concepts to Understand

### Epistemic vs Aleatoric Uncertainty
- **Epistemic**: Model uncertainty (reducible with more data)
- **Aleatoric**: Data uncertainty (irreducible noise)

### When to Use Each Method?
1. **MC Dropout**: Quick uncertainty with single model
2. **Deep Ensembles**: Best quality, but expensive
3. **Temperature Scaling**: Calibration fix for any model
4. **Conformal Prediction**: Guaranteed coverage for safety-critical apps

---

## 🎯 Implementation Details Below

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```

# Day 1-2: MC Dropout 구현

```
class GCN_with_Dropout(nn.Module):
    """
    Monte Carlo Dropout을 위한 GNN
    핵심: 추론 시에도 dropout을 켜둠!
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index, training=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 핵심: training=True면 항상 dropout 적용
        x = F.dropout(x, p=self.dropout, training=training or self.training)
        x = self.conv2(x, edge_index)
        return x


def mc_dropout_prediction(model, data, n_samples=50):
    """
    MC Dropout으로 uncertainty 측정
    
    Args:
        model: GCN_with_Dropout 모델
        data: 그래프 데이터
        n_samples: Dropout sampling 횟수
    
    Returns:
        mean_pred: 평균 예측 (N, C)
        epistemic_uncertainty: Epistemic 불확실성 (N,)
        entropy: Predictive entropy (N,)
    """
    model.eval()
    all_predictions = []
    
    # n_samples번 forward pass (매번 다른 dropout mask)
    with torch.no_grad():
        for _ in range(n_samples):
            # training=True로 설정하여 dropout 활성화
            logits = model(data.x, data.edge_index, training=True)
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs)
    
    # (n_samples, num_nodes, num_classes) -> (num_nodes, num_classes)
    all_predictions = torch.stack(all_predictions)
    mean_pred = all_predictions.mean(dim=0)
    
    # Epistemic Uncertainty: Variance across samples
    epistemic = all_predictions.var(dim=0).mean(dim=1)
    
    # Predictive Entropy
    entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
    
    return mean_pred, epistemic, entropy
```


# Day 3-4: Deep Ensembles 구현

```
class GCN_Ensemble:
    """
    Deep Ensemble: 여러 모델을 독립적으로 학습
    각 모델은 다른 random seed로 초기화
    """
    def __init__(self, in_channels, hidden_channels, out_channels, n_models=5):
        self.models = []
        self.n_models = n_models
        
        for i in range(n_models):
            # 각 모델마다 다른 seed
            torch.manual_seed(42 + i)
            model = GCNConv_Model(in_channels, hidden_channels, out_channels)
            self.models.append(model)
    
    def train_ensemble(self, data, epochs=200):
        """각 모델을 독립적으로 학습"""
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 50 == 0:
                    val_acc = self.evaluate_single(model, data, data.val_mask)
                    print(f"  Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def predict(self, data):
        """
        Ensemble 예측 및 불확실성 측정
        
        Returns:
            mean_pred: 평균 예측
            epistemic: 모델 간 disagreement
            entropy: Predictive entropy
        """
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                probs = F.softmax(logits, dim=1)
                all_predictions.append(probs)
        
        all_predictions = torch.stack(all_predictions)  # (n_models, N, C)
        mean_pred = all_predictions.mean(dim=0)
        
        # Epistemic: 모델들의 disagreement
        epistemic = all_predictions.var(dim=0).mean(dim=1)
        
        # Entropy
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
        
        return mean_pred, epistemic, entropy
    
    def evaluate_single(self, model, data, mask):
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            acc = (pred[mask] == data.y[mask]).float().mean()
        return acc


# Helper class
class GCNConv_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
```


# Day 5: Temperature Scaling (Calibration)

```
class TemperatureScaling(nn.Module):
    """
    Temperature Scaling으로 probability calibration
    
    학습된 모델의 logits를 temperature로 나눠서 보정
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        """
        logits: (N, C) - 모델의 raw output
        return: (N, C) - temperature-scaled probabilities
        """
        return F.softmax(logits / self.temperature, dim=1)
    
    def calibrate(self, model, data, val_mask, max_iter=50):
        """
        Validation set으로 optimal temperature 찾기
        NLL을 최소화하는 temperature를 학습
        """
        # Get validation logits
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_logits = logits[val_mask]
            val_labels = data.y[val_mask]
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()
```


# Day 6-7: Conformal Prediction

```
class ConformalPredictor:
    """
    Conformal Prediction: Distribution-free uncertainty
    
    핵심 아이디어:
    - Calibration set에서 nonconformity score 계산
    - Test time에 prediction set 생성 (guaranteed coverage)
    """
    def __init__(self, alpha=0.1):
        """
        alpha: 유의수준 (1-alpha = coverage level)
        alpha=0.1이면 90% coverage 보장
        """
        self.alpha = alpha
        self.quantile = None
    
    def calibrate(self, model, data, cal_mask):
        """
        Calibration set에서 nonconformity scores 계산
        
        Nonconformity score: 1 - P(y_true)
        즉, 정답 클래스의 확률이 낮을수록 높은 score
        """
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)
            
            cal_probs = probs[cal_mask]
            cal_labels = data.y[cal_mask]
            
            # Nonconformity scores
            scores = 1 - cal_probs[torch.arange(len(cal_labels)), cal_labels]
            
            # (1-alpha) quantile 계산
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.quantile = torch.quantile(scores, q_level)
            
        print(f"Conformal quantile (alpha={self.alpha}): {self.quantile:.4f}")
        return self.quantile
    
    def predict(self, model, data, test_mask):
        """
        Prediction sets 생성
        
        Returns:
            prediction_sets: List of sets, 각 노드마다 가능한 클래스들
            set_sizes: 각 prediction set의 크기
        """
        if self.quantile is None:
            raise ValueError("먼저 calibrate()를 호출하세요!")
        
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)
            test_probs = probs[test_mask]
            
            # Prediction set: {y : 1 - P(y) <= quantile}
            # 즉, P(y) >= 1 - quantile인 모든 클래스
            threshold = 1 - self.quantile
            prediction_sets = (test_probs >= threshold).cpu().numpy()
            set_sizes = prediction_sets.sum(axis=1)
            
        return prediction_sets, set_sizes
    
    def evaluate_coverage(self, model, data, test_mask):
        """
        Coverage 측정: 정답이 prediction set에 포함된 비율
        이론적으로 (1-alpha) 이상이어야 함
        """
        prediction_sets, set_sizes = self.predict(model, data, test_mask)
        test_labels = data.y[test_mask].cpu().numpy()
        
        coverage = np.mean([prediction_sets[i, test_labels[i]] 
                           for i in range(len(test_labels))])
        avg_set_size = np.mean(set_sizes)
        
        print(f"Coverage: {coverage:.4f} (target: {1-self.alpha:.4f})")
        print(f"Average prediction set size: {avg_set_size:.2f}")
        
        return coverage, avg_set_size
```


# Evaluation Metrics for UQ

```
def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE)
    
    Confidence와 accuracy가 얼마나 일치하는지 측정
    ECE가 낮을수록 well-calibrated
    """
    confidences = probs.max(dim=1)[0].cpu().numpy()
    predictions = probs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Bin에 속하는 샘플들
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        
        if in_bin.sum() > 0:
            bin_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            bin_confidence = confidences[in_bin].mean()
            ece += (in_bin.sum() / len(labels)) * abs(bin_accuracy - bin_confidence)
    
    return ece


def compute_nll(probs, labels):
    """
    Negative Log-Likelihood
    
    확률 예측의 품질 측정
    NLL이 낮을수록 좋은 확률 예측
    """
    labels = labels.cpu()
    probs = probs.cpu()
    nll = -torch.log(probs[torch.arange(len(labels)), labels] + 1e-10).mean()
    return nll.item()


def compute_brier_score(probs, labels, num_classes):
    """
    Brier Score: 확률 예측의 정확도
    
    낮을수록 좋음 (0이 perfect)
    """
    labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
    brier = ((probs - labels_onehot) ** 2).sum(dim=1).mean()
    return brier.item()
```


# Visualization Functions

```
def plot_reliability_diagram(probs, labels, n_bins=10, title="Reliability Diagram"):
    """
    Calibration plot: Confidence vs Accuracy
    대각선에 가까울수록 well-calibrated
    """
    confidences = probs.max(dim=1)[0].cpu().numpy()
    predictions = probs.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_confidences.append(confidences[in_bin].mean())
            bin_accuracies.append((predictions[in_bin] == labels[in_bin]).mean())
            bin_counts.append(in_bin.sum())
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.bar(bin_confidences, bin_accuracies, width=1/n_bins, alpha=0.7, 
            edgecolor='black', label='Model')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reliability_diagram.png', dpi=150)
    plt.show()


def plot_uncertainty_vs_error(uncertainty, is_correct, title="Uncertainty vs Error"):
    """
    불확실성이 높은 샘플이 틀릴 확률이 높은가?
    """
    uncertainty = uncertainty.cpu().numpy()
    is_correct = is_correct.cpu().numpy()
    
    # Bin by uncertainty
    n_bins = 10
    bins = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_error_rates = []
    bin_centers = []
    
    for i in range(n_bins):
        in_bin = (uncertainty >= bins[i]) & (uncertainty < bins[i+1])
        if in_bin.sum() > 0:
            error_rate = 1 - is_correct[in_bin].mean()
            bin_error_rates.append(error_rate)
            bin_centers.append((bins[i] + bins[i+1]) / 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, bin_error_rates, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Uncertainty (binned)', fontsize=14)
    plt.ylabel('Error Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('uncertainty_vs_error.png', dpi=150)
    plt.show()
```


# Week 2 Complete Example

```
def week2_complete_pipeline():
    """
    Week 2 전체 파이프라인:
    4가지 UQ 방법을 모두 비교
    """
    print("=" * 60)
    print("Week 2: Uncertainty Quantification Complete Pipeline")
    print("=" * 60)
    
    # 데이터 로드
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    
    results = {}
    
    # ===== 1. MC Dropout =====
    print("\n[1/4] Training MC Dropout model...")
    mc_model = GCN_with_Dropout(
        dataset.num_node_features, 16, dataset.num_classes, dropout=0.5
    )
    # ... training code ...
    
    print("Predicting with MC Dropout (50 samples)...")
    mc_probs, mc_epistemic, mc_entropy = mc_dropout_prediction(mc_model, data, n_samples=50)
    
    results['MC Dropout'] = {
        'probs': mc_probs,
        'epistemic': mc_epistemic,
        'entropy': mc_entropy
    }
    
    # ===== 2. Deep Ensembles =====
    print("\n[2/4] Training Deep Ensemble (5 models)...")
    ensemble = GCN_Ensemble(dataset.num_node_features, 16, dataset.num_classes, n_models=5)
    ensemble.train_ensemble(data, epochs=200)
    
    print("Predicting with Ensemble...")
    ens_probs, ens_epistemic, ens_entropy = ensemble.predict(data)
    
    results['Ensemble'] = {
        'probs': ens_probs,
        'epistemic': ens_epistemic,
        'entropy': ens_entropy
    }
    
    # ===== 3. Temperature Scaling =====
    print("\n[3/4] Calibrating with Temperature Scaling...")
    temp_scaler = TemperatureScaling()
    temp_scaler.calibrate(mc_model, data, data.val_mask)
    
    # ===== 4. Conformal Prediction =====
    print("\n[4/4] Computing Conformal Prediction Sets...")
    conformal = ConformalPredictor(alpha=0.1)
    conformal.calibrate(mc_model, data, data.val_mask)
    coverage, avg_set_size = conformal.evaluate_coverage(mc_model, data, data.test_mask)
    
    # ===== Comparison =====
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON (Test Set)")
    print("=" * 60)
    
    for method_name, result in results.items():
        probs = result['probs'][data.test_mask]
        labels = data.y[data.test_mask]
        
        # Accuracy
        acc = (probs.argmax(dim=1) == labels).float().mean()
        
        # UQ Metrics
        ece = compute_ece(probs, labels)
        nll = compute_nll(probs, labels)
        brier = compute_brier_score(probs, labels, dataset.num_classes)
        
        print(f"\n{method_name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ECE: {ece:.4f}")
        print(f"  NLL: {nll:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Avg Epistemic: {result['epistemic'][data.test_mask].mean():.4f}")
    
    print(f"\nConformal Prediction:")
    print(f"  Coverage: {coverage:.4f} (target: 0.90)")
    print(f"  Avg Set Size: {avg_set_size:.2f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_reliability_diagram(mc_probs[data.test_mask], data.y[data.test_mask], 
                            title="MC Dropout Calibration")
    
    is_correct = (mc_probs.argmax(dim=1) == data.y)[data.test_mask]
    plot_uncertainty_vs_error(mc_epistemic[data.test_mask], is_correct,
                             title="MC Dropout: Uncertainty vs Error")
    
    return results
```


# ✅ Week 2 완료 체크리스트

## 1. 논문 이해 (Paper Reading)
- [ ] MC Dropout 논문 읽고 요약
- [ ] Deep Ensembles 논문 읽고 요약
- [ ] Temperature Scaling 논문 읽고 요약
- [ ] Conformal Prediction tutorial 읽고 요약

## 2. 코드 구현 (Implementation)
- [ ] 4가지 UQ 방법 모두 구현 완료
- [ ] Cora에서 실험 완료
- [ ] ECE, NLL, Brier Score 계산 가능

## 3. 실험 (Experiments)
- [ ] MC Dropout: n_samples 영향 실험 (10, 50, 100)
- [ ] Ensemble: n_models 영향 실험 (3, 5, 10)
- [ ] Reliability diagram 생성
- [ ] Uncertainty vs Error plot 생성

## 4. 이해도 (Conceptual Understanding)
- [ ] Epistemic vs Aleatoric 차이 설명 가능
- [ ] "언제 어떤 UQ 방법을 쓰나?" 답변 가능
- [ ] Calibration의 중요성 설명 가능

---

**모두 체크되면 Week 3 (GNN + UQ)로! 🚀**

if __name__ == "__main__":
    week2_complete_pipeline()