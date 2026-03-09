# Neural Relational Inference for Interacting Systems
## Kipf et al., ICML 2018

### One-Paragraph Summary

Neural Relational Inference (NRI) is an unsupervised method for simultaneously discovering interaction structures and learning dynamical models from observational trajectory data. The model uses a variational autoencoder (VAE) framework where the latent code represents a discrete interaction graph between objects, and reconstruction is performed via graph neural networks. Given only observed trajectories (e.g., positions and velocities of particles), NRI infers which objects interact and how, without requiring ground-truth labels. The model achieves near-perfect accuracy (99.9%) on recovering ground-truth interactions in simulated physics systems (springs, charged particles, Kuramoto oscillators) and learns interpretable structures on real-world motion capture and NBA basketball tracking data.

### Key Method: Encoder-Decoder with Graph Learning

**Architecture:**
- **Encoder**: GNN that maps trajectories to edge-type distributions q_φ(z|x)
  - Multiple rounds of node-to-edge (v→e) and edge-to-node (e→v) message passing
  - Output: softmax distribution over K edge types for each pair (i,j)
  - Uses MLPs or 1D CNNs with attention as message functions

- **Decoder**: GNN that predicts future dynamics p_θ(x|z) conditioned on sampled graph
  - Separate MLP for each edge type k
  - Messages weighted by sampled edge types z_{ij,k}
  - Predicts state changes Δx (position/velocity differences)

**VAE Objective (ELBO):**
```
L = E_{q_φ(z|x)}[log p_θ(x|z)] - KL[q_φ(z|x)||p_θ(z)]
```
- Reconstruction: MSE between predicted and true trajectories
- KL term: Encourages sparsity via uniform prior with "non-edge" option

**Discrete Sampling:**
- Gumbel-softmax (concrete distribution) for differentiable sampling of discrete edge types
- Temperature τ controls smoothness; converges to categorical as τ→0

**Key Innovation - Avoiding Degenerate Decoders:**
- Multi-step prediction (M=10 steps) to force model to use edge information
- Separate MLP per edge type makes edge dependence explicit

### Experimental Results

| Dataset | Edge Accuracy | Notes |
|---------|---------------|-------|
| Springs (5 obj) | 99.9% | Near supervised performance |
| Charged (5 obj) | 82.1% | Weak interactions harder |
| Kuramoto (5 obj) | 96.0% | Phase-coupled oscillators |
| Motion capture | N/A | Learns hand-to-extremity connections |
| NBA tracking | N/A | Separates ball/handler from others |

### Limitations

1. **Static graph assumption**: Model learns fixed interaction graphs during training; dynamic re-estimation needed at test time for time-varying interactions

2. **Scalability**: Fully-connected graph assumption means O(N²) edges; not practical for large N

3. **Discrete edge types**: Requires pre-specifying number of edge types K; may not capture continuous interaction strengths

4. **Simulation focus**: Primarily validated on physics simulations where ground truth is available; real-world validation limited to qualitative assessment

5. **No temporal structure change**: Cannot model regime changes where the interaction graph fundamentally shifts over time

6. **Supervision-free but requires trajectory data**: Needs clean trajectory observations; cannot work with aggregate or noisy financial data directly

### How Our Work Differs

| Aspect | NRI (Kipf et al.) | Our Work |
|--------|-------------------|----------|
| **Domain** | Physics simulations, sports tracking | Financial factor returns |
| **Graph type** | Discrete edge types (K categories) | Continuous Granger causality strengths |
| **Temporal** | Static graph (within trajectory) | Regime-dependent, time-varying graphs |
| **Regime changes** | Not modeled | Core focus (HMM for regime detection) |
| **Structural breaks** | Not considered | Explicit breakpoint detection (sup-F test) |
| **Inference** | VAE with Gumbel-softmax | Classical Granger + HMM, no neural networks |
| **Predictability decay** | Not studied | Main contribution (half-life = 3.35 years) |
| **Validation** | Ground-truth recovery | OOS prediction, international replication |

**Key Distinction**: NRI discovers *what* interactions exist in a stationary dynamical system. Our work studies *how* interaction structures (cross-factor predictability) change over time and eventually decay, framing this as a regime-switching process with detectable structural breaks.

### Relevance to Our Paper

NRI represents the neural network approach to relational structure discovery. While methodologically different from our Granger causality + HMM framework, it addresses a related problem: inferring latent interaction graphs from observational data. The key insight from NRI relevant to our work is that interaction structures can be treated as latent variables to be inferred rather than assumed.

However, NRI's assumption of static graphs within trajectories makes it less suitable for financial applications where regime changes are fundamental. Our contribution of documenting predictability decay over decades-long horizons addresses a temporal scale that NRI was not designed for.

### Citation

```bibtex
@inproceedings{kipf2018neural,
  title={Neural Relational Inference for Interacting Systems},
  author={Kipf, Thomas and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  booktitle={International Conference on Machine Learning},
  pages={2688--2697},
  year={2018},
  organization={PMLR}
}
```
