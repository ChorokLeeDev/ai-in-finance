"""
Run ensemble of LightGBM models for REGRESSION tasks to analyze uncertainty quantification.

This script is specifically designed for regression tasks where epistemic uncertainty
is measured as the standard deviation across ensemble predictions.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict

os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything

from relbench.base import EntityTask, TaskType
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble of LightGBM models for regression UQ"
    )
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="results-position")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of trials for hyperparameter tuning")
    parser.add_argument("--sample_size", type=int, default=50_000,
                        help="Subsample training data size")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
                        help="List of random seeds for ensemble")
    parser.add_argument("--cache_dir", type=str,
                        default=os.path.expanduser("~/.cache/relbench_examples"))
    parser.add_argument("--download", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # Load task
    task: EntityTask = get_task(args.dataset, args.task, download=args.download)
    dataset = task.dataset

    if task.task_type != TaskType.REGRESSION:
        raise ValueError(f"This script is for REGRESSION tasks only. "
                         f"Got {task.task_type}")

    print(f"\n{'='*60}")
    print(f"Task: {args.dataset} / {args.task}")
    print(f"Task type: {task.task_type}")
    print(f"Training ensemble with {len(args.seeds)} models")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}\n")

    # Train models with different seeds
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training model with seed {seed}")
        print(f"{'='*60}\n")

        seed_everything(seed)
        np.random.seed(seed)

        # Load tables
        train_table = task.get_table("train")
        val_table = task.get_table("val")
        test_table = task.get_table("test", mask_input_cols=False)

        # Get stypes
        entity_table = dataset.get_db().table_dict[task.entity_table]
        entity_df = entity_table.df

        stypes_cache_path = Path(
            f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/stypes.json"
        )
        try:
            with open(stypes_cache_path, "r") as f:
                col_to_stype_dict = json.load(f)
            for table, col_to_stype in col_to_stype_dict.items():
                for col, stype_str in col_to_stype.items():
                    col_to_stype[col] = stype(stype_str)
        except FileNotFoundError:
            col_to_stype_dict = get_stype_proposal(dataset.get_db())
            Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stypes_cache_path, "w") as f:
                json.dump(col_to_stype_dict, f, indent=2, default=str)

        col_to_stype = col_to_stype_dict[task.entity_table]
        remove_pkey_fkey(col_to_stype, entity_table)
        for col in dataset.remove_columns:
            if col in col_to_stype:
                del col_to_stype[col]

        # Set target column type for regression
        col_to_stype[task.target_col] = torch_frame.numerical

        # Subsample training data
        sampled_idx = None
        if 0 < args.sample_size < len(train_table):
            sampled_idx = np.random.permutation(len(train_table))[:args.sample_size]
            train_table.df = train_table.df.iloc[sampled_idx]

        # Prepare dataframes
        dfs: Dict[str, pd.DataFrame] = {}
        for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
            left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
            entity_df = entity_df.astype({entity_table.pkey_col: table.df[left_entity].dtype})

            # Remove duplicated columns
            for col in set(entity_df.columns).intersection(set(table.df.columns)):
                if col != entity_table.pkey_col:
                    entity_df = entity_df.drop(columns=[col])

            dfs[split] = table.df.merge(
                entity_df,
                how="left",
                left_on=left_entity,
                right_on=entity_table.pkey_col,
            )

        # Create torch_frame dataset
        train_dataset = torch_frame.data.Dataset(
            df=dfs["train"],
            col_to_stype=col_to_stype,
            target_col=task.target_col,
            col_to_text_embedder_cfg=TextEmbedderConfig(
                text_embedder=GloveTextEmbedding(device=device),
                batch_size=256,
            ),
        )

        path = Path(
            f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized/"
            f"node_train_regression_{args.sample_size}_seed{seed}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        train_dataset = train_dataset.materialize(path=path)

        tf_train = train_dataset.tensor_frame
        tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
        tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

        # Train LightGBM
        model = LightGBM(
            task_type=train_dataset.task_type,
            metric=Metric.MAE,
        )
        model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=args.num_trials)

        # Get predictions
        train_pred = model.predict(tf_test=tf_train).numpy()
        val_pred = model.predict(tf_test=tf_val).numpy()
        test_pred = model.predict(tf_test=tf_test).numpy()

        # Evaluate
        train_table_orig = task.get_table("train")
        if sampled_idx is not None:
            train_table_orig.df = train_table_orig.df.iloc[sampled_idx]

        train_metrics = task.evaluate(train_pred, train_table_orig)
        val_metrics = task.evaluate(val_pred, val_table)
        test_metrics = task.evaluate(test_pred)

        print(f"\nResults for seed {seed}:")
        print(f"  Train: {train_metrics}")
        print(f"  Val: {val_metrics}")
        print(f"  Test: {test_metrics}")

        # Save predictions
        results_dir = Path(f"results/ensemble/{args.dataset}/{args.task}")
        results_dir.mkdir(parents=True, exist_ok=True)

        save_data = {
            'seed': seed,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred,
            'train_true': train_table_orig.df[task.target_col].values,
            'val_true': val_table.df[task.target_col].values,
            'test_true': test_table.df[task.target_col].values,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'task_type': str(task.task_type),
        }

        save_path = results_dir / f"seed_{seed}_sample_{args.sample_size}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved predictions to: {save_path}")

    print(f"\n{'='*60}")
    print(f"All {len(args.seeds)} models trained successfully!")
    print(f"Results saved to: results/ensemble/{args.dataset}/{args.task}/")
    print(f"\nNext step: Run analysis with:")
    print(f"  python analyze_regression_uq.py --dataset {args.dataset} --task {args.task}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
