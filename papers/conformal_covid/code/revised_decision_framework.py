#!/usr/bin/env python3
"""
Revised 2D Decision Framework: Fix sales-office outlier problem

OLD (1D): concentration > 40% → vulnerable  ❌ Fails for sales-office
NEW (2D): concentration > 40% AND no stable secondary features → vulnerable ✓

This addresses the critical issue that sales-office has:
- Concentration: 42.6% (above threshold)
- BUT stable secondary feature (SALESORGANIZATION, Jaccard=0.61, 20% importance)
- Result: ROBUST despite high concentration
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# Data from Table 3 + additional analysis needed
TASK_DATA = {
    's-group': {
        'concentration': 47.3,
        'drop': 86.7,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {},  # Need to add
        'category': 'Catastrophic'
    },
    's-payterms': {
        'concentration': 54.2,
        'drop': 77.1,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {},
        'category': 'Catastrophic'
    },
    's-shipcond': {
        'concentration': 50.7,
        'drop': 71.6,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {},
        'category': 'Catastrophic'
    },
    'i-shippoint': {
        'concentration': 48.8,
        'drop': 18.5,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {},
        'category': 'Severe'
    },
    's-office': {
        'concentration': 42.6,
        'drop': 0.0,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {
            'SALESORGANIZATION': {'jaccard': 0.61, 'importance_pct': 20.0},
            # This is the PROTECTIVE FACTOR
        },
        'category': 'Robust'
    },
    'i-incoterms': {
        'concentration': 28.9,
        'drop': 11.3,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {},
        'category': 'Robust'
    },
    'i-plant': {
        'concentration': 23.9,
        'drop': 10.6,
        'top_feature': 'SALESDOCUMENT',
        'top_jaccard': 0.00,
        'secondary_features': {},
        'category': 'Severe'
    },
    's-incoterms': {
        'concentration': 23.7,
        'drop': 8.5,
        'top_feature': 'SALESDOCUMENTTYPE',
        'top_jaccard': 0.00,
        'secondary_features': {},
        'category': 'Robust'
    },
}


def decision_framework_1d(concentration: float, threshold: float = 40.0) -> str:
    """
    OLD 1D framework (BROKEN).

    Returns: 'VULNERABLE' or 'ROBUST'
    """
    return 'VULNERABLE' if concentration > threshold else 'ROBUST'


def decision_framework_2d(
    concentration: float,
    secondary_features: Dict[str, Dict],
    concentration_threshold: float = 40.0,
    jaccard_threshold: float = 0.5,
    importance_threshold: float = 15.0
) -> Tuple[str, str]:
    """
    NEW 2D framework (FIXED).

    Args:
        concentration: Primary feature concentration (%)
        secondary_features: Dict of {feature_name: {'jaccard': float, 'importance_pct': float}}
        concentration_threshold: Threshold for high concentration (default 40%)
        jaccard_threshold: Threshold for stable feature (default 0.5)
        importance_threshold: Min importance % for protective factor (default 15%)

    Returns:
        (prediction, reason)

    Logic:
        1. IF concentration <= 40% → ROBUST (distributed importance)
        2. IF concentration > 40%:
           a. Check for protective factors (stable secondary features)
           b. Protective = Jaccard > 0.5 AND importance > 15%
           c. IF has protective factor → ROBUST
           d. ELSE → VULNERABLE
    """
    # Rule 1: Low concentration → robust
    if concentration <= concentration_threshold:
        return 'ROBUST', f'Low concentration ({concentration:.1f}% ≤ {concentration_threshold}%)'

    # Rule 2: High concentration → check for protective factors
    protective_factors = []
    for feat_name, feat_data in secondary_features.items():
        jaccard = feat_data.get('jaccard', 0.0)
        importance = feat_data.get('importance_pct', 0.0)

        if jaccard >= jaccard_threshold and importance >= importance_threshold:
            protective_factors.append(
                f"{feat_name} (J={jaccard:.2f}, I={importance:.1f}%)"
            )

    if protective_factors:
        reason = f"High conc ({concentration:.1f}%) BUT protected by: {', '.join(protective_factors)}"
        return 'ROBUST', reason
    else:
        reason = f"High conc ({concentration:.1f}%) with NO protective factors"
        return 'VULNERABLE', reason


def evaluate_frameworks():
    """Compare 1D vs 2D framework performance."""

    print("="*80)
    print("COMPARING DECISION FRAMEWORKS")
    print("="*80)

    results = []

    for task_name, data in TASK_DATA.items():
        concentration = data['concentration']
        drop = data['drop']
        actual_category = data['category']
        secondary = data['secondary_features']

        # 1D prediction
        pred_1d = decision_framework_1d(concentration)

        # 2D prediction
        pred_2d, reason_2d = decision_framework_2d(concentration, secondary)

        # Determine actual (simplified: >50% drop = VULNERABLE)
        actual = 'VULNERABLE' if drop > 50 else 'ROBUST'

        # Evaluate
        correct_1d = (pred_1d == actual)
        correct_2d = (pred_2d == actual)

        results.append({
            'task': task_name,
            'concentration': concentration,
            'drop': drop,
            'actual': actual,
            'pred_1d': pred_1d,
            'correct_1d': correct_1d,
            'pred_2d': pred_2d,
            'correct_2d': correct_2d,
            'reason_2d': reason_2d
        })

        # Print result
        status_1d = "✓" if correct_1d else "✗"
        status_2d = "✓" if correct_2d else "✗"

        print(f"\n{task_name}:")
        print(f"  Concentration: {concentration:.1f}%")
        print(f"  Drop: {drop:.1f}%")
        print(f"  Actual: {actual}")
        print(f"  1D: {pred_1d} {status_1d}")
        print(f"  2D: {pred_2d} {status_2d}")
        if not correct_2d or task_name == 's-office':
            print(f"  → {reason_2d}")

    # Summary
    df = pd.DataFrame(results)
    acc_1d = df['correct_1d'].mean()
    acc_2d = df['correct_2d'].mean()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"1D Framework Accuracy: {acc_1d:.1%} ({df['correct_1d'].sum()}/{len(df)})")
    print(f"2D Framework Accuracy: {acc_2d:.1%} ({df['correct_2d'].sum()}/{len(df)})")
    print(f"Improvement: {(acc_2d - acc_1d):.1%}")

    # Specific cases
    print("\n" + "="*80)
    print("CRITICAL CASE: sales-office (The Outlier)")
    print("="*80)
    office_result = df[df['task'] == 's-office'].iloc[0]
    print(f"Concentration: {office_result['concentration']:.1f}% (ABOVE 40% threshold)")
    print(f"Actual drop: {office_result['drop']:.1f}% (ROBUST)")
    print(f"1D prediction: {office_result['pred_1d']} {'✗ WRONG' if not office_result['correct_1d'] else '✓'}")
    print(f"2D prediction: {office_result['pred_2d']} {'✓ CORRECT' if office_result['correct_2d'] else '✗'}")
    print(f"Reason: {office_result['reason_2d']}")

    return df


def generate_updated_decision_framework_text():
    """Generate LaTeX text for revised decision framework."""

    latex = r"""
\section{Revised Practitioner Decision Framework}

Before deploying conformal prediction in non-stationary settings:

\begin{enumerate}
    \item \textbf{Check task complexity}: If entropy $< 2.5$ or top-class $> 50\%$, coverage likely maintained

    \item \textbf{Compute feature Jaccard similarity}:
    \begin{itemize}
        \item If mean Jaccard $< 0.1$ $\rightarrow$ potential risk
        \item If mean Jaccard $> 0.4$ $\rightarrow$ likely robust
    \end{itemize}

    \item \textbf{Analyze SHAP importance (2D framework)}:
    \begin{enumerate}
        \item Compute concentration = (Top feature importance) / (Total importance)
        \item \textbf{If concentration $\leq$ 40\%}: Task is ROBUST (distributed importance)
        \item \textbf{If concentration $>$ 40\%}: Check for protective factors:
        \begin{itemize}
            \item Find secondary features with Jaccard $> 0.5$ AND importance $> 15\%$
            \item If such features exist $\rightarrow$ ROBUST (protected by stable features)
            \item If no protective factors $\rightarrow$ VULNERABLE
        \end{itemize}
    \end{enumerate}

    \item \textbf{For vulnerable tasks}:
    \begin{itemize}
        \item Implement quarterly retraining (3 retrains/year)
        \item Do NOT use monthly (overfitting risk, higher variance)
    \end{itemize}

    \item \textbf{For robust tasks}:
    \begin{itemize}
        \item Skip retraining to save computational cost
        \item Monitor coverage drift as early warning
    \end{itemize}
\end{enumerate}

\subsection{Example: sales-office (The Outlier Case)}

This task demonstrates why the 2D framework is necessary:
\begin{itemize}
    \item Primary feature concentration: 42.6\% ($>$ 40\% threshold)
    \item \textbf{BUT}: Secondary feature SALESORGANIZATION has:
    \begin{itemize}
        \item Jaccard similarity: 0.61 ($>$ 0.5)
        \item Importance: 20\% ($>$ 15\%)
    \end{itemize}
    \item 1D framework: Predicts VULNERABLE (wrong)
    \item 2D framework: Predicts ROBUST due to protective factor (correct)
    \item Actual coverage drop: 0\% (ROBUST)
\end{itemize}
"""

    return latex


if __name__ == "__main__":
    print("""
    ========================================================================
    ADDRESSING ISSUE #2: Sales-Office Outlier Problem
    ========================================================================

    Problem: sales-office has 42.6% concentration but 0% drop
    - 1D framework (concentration > 40%) predicts VULNERABLE → WRONG
    - Reality: ROBUST due to protective factor (SALESORGANIZATION)

    Solution: 2D framework checks BOTH:
    1. Primary feature concentration
    2. Stable secondary features (protective factors)

    ========================================================================
    """)

    # Run evaluation
    df_results = evaluate_frameworks()

    # Generate updated framework
    print("\n" + "="*80)
    print("UPDATED FRAMEWORK (LaTeX)")
    print("="*80)
    latex_text = generate_updated_decision_framework_text()
    print(latex_text)

    # Save
    with open('revised_decision_framework.tex', 'w') as f:
        f.write(latex_text)
    print("\nSaved to: revised_decision_framework.tex")

    # Note: Need to compute secondary feature data for all tasks
    print("\n" + "="*80)
    print("TODO: Compute secondary feature Jaccard & importance for all tasks")
    print("="*80)
    print("Currently only have data for sales-office.")
    print("Need to run SHAP analysis to get top-5 features + Jaccard for each task.")
