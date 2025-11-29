"""
Data Loader for rel-stack Dataset (FK-Level Risk Attribution)
================================================================

rel-stack schema:
- posts: Id (pkey), OwnerUserId, PostTypeId, Title, Tags
- users: Id (pkey), DisplayName, Location, CreationDate
- votes: PostId (fkey→posts), UserId (fkey→users), VoteTypeId
- comments: PostId (fkey→posts), UserId (fkey→users), Text
- badges: UserId (fkey→users), Class, Name
- postHistory: PostId (fkey→posts), UserId (fkey→users)
- postLinks: PostId (fkey→posts), RelatedPostId (fkey→posts)

FK Groups:
- POST: post attributes (title, type, tags)
- USER: owner/author attributes
- ENGAGEMENT: votes, comments, links
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from cache import save_cache, load_cache


def get_fk_group(col_name: str) -> str:
    """Map column name to FK group."""
    col_lower = col_name.lower()

    # Post features
    if any(x in col_lower for x in ['post', 'title', 'tags', 'type', 'body', 'score', 'answer']):
        return 'POST'

    # User features
    if any(x in col_lower for x in ['user', 'owner', 'display', 'reputation', 'location', 'account']):
        return 'USER'

    # Engagement features (votes, comments, links)
    if any(x in col_lower for x in ['vote', 'comment', 'link', 'view', 'favorite']):
        return 'ENGAGEMENT'

    # Default to POST
    return 'POST'


def load_stack_data(
    task_name: str = 'post-votes',
    sample_size: int = 3000,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, str]]:
    """Load rel-stack data for FK-level risk attribution."""

    cache_key = f"data_stack_{task_name}_{sample_size}"

    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            print(f"[CACHE] Loaded: {cache_key}.pkl")
            return (
                cached['X'],
                cached['y'],
                cached['feature_cols'],
                cached['col_to_fk']
            )

    print(f"[DATA] Loading rel-stack {task_name}...")

    # Load dataset and task
    dataset = get_dataset('rel-stack', download=True)
    task = get_task('rel-stack', task_name, download=True)
    db = dataset.get_db()

    # Get tables
    posts_df = db.table_dict['posts'].df.copy()
    users_df = db.table_dict['users'].df.copy()
    votes_df = db.table_dict['votes'].df.copy()
    comments_df = db.table_dict['comments'].df.copy()

    # Get train table
    train_table = task.get_table('train')
    train_df = train_table.df.copy()

    print(f"[DATA] Train size: {len(train_df)}")

    # Sample first for efficiency
    if len(train_df) > sample_size * 2:
        train_df = train_df.sample(n=sample_size * 2, random_state=42)

    # Join with posts
    merged = train_df.merge(
        posts_df[['Id', 'OwnerUserId', 'PostTypeId']].rename(columns={'Id': 'PostId'}),
        on='PostId',
        how='left'
    )

    # Create post features
    merged['PostTypeId'] = merged['PostTypeId'].fillna(0).astype(int)

    # Join with users (post owner)
    user_features = users_df[['Id', 'AccountId']].copy()
    user_features['AccountId'] = user_features['AccountId'].fillna(0)
    merged = merged.merge(
        user_features.rename(columns={'Id': 'OwnerUserId', 'AccountId': 'OwnerAccountId'}),
        on='OwnerUserId',
        how='left'
    )

    # Aggregate vote features per post
    vote_agg = votes_df.groupby('PostId').agg({
        'Id': 'count',
        'VoteTypeId': ['mean', 'std']
    }).reset_index()
    vote_agg.columns = ['PostId', 'vote_count', 'vote_type_mean', 'vote_type_std']
    vote_agg['vote_type_std'] = vote_agg['vote_type_std'].fillna(0)
    merged = merged.merge(vote_agg, on='PostId', how='left')

    # Aggregate comment features per post
    comment_agg = comments_df.groupby('PostId').agg({
        'Id': 'count'
    }).reset_index()
    comment_agg.columns = ['PostId', 'comment_count']
    merged = merged.merge(comment_agg, on='PostId', how='left')

    # Fill NaN and convert to numeric
    for col in merged.columns:
        if col not in ['PostId', 'timestamp', 'OwnerUserId']:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    # Final sample
    if len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=42)

    # Define features
    target_col = task.target_col
    exclude_cols = [target_col, 'PostId', 'timestamp', 'OwnerUserId']
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    X = merged[feature_cols].copy()
    y = merged[target_col].copy()

    # Map to FK groups
    col_to_fk = {col: get_fk_group(col) for col in feature_cols}

    print(f"\n[DATA] Final shape: X={X.shape}, y={len(y)}")
    print(f"[DATA] FK groups: {set(col_to_fk.values())}")

    # Print FK summary
    print_fk_summary(col_to_fk)

    # Cache
    if use_cache:
        save_cache(cache_key, {
            'X': X,
            'y': y,
            'feature_cols': feature_cols,
            'col_to_fk': col_to_fk
        })

    return X, y, feature_cols, col_to_fk


def print_fk_summary(col_to_fk: Dict[str, str]):
    """Print FK group summary."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    print("\n--- FK Groups ---")
    for fk, cols in sorted(fk_to_cols.items()):
        print(f"  {fk}: {cols}")


if __name__ == "__main__":
    print("Testing rel-stack data loading...")

    X, y, feature_cols, col_to_fk = load_stack_data(
        task_name='post-votes',
        sample_size=1000,
        use_cache=False
    )

    print(f"\nFeatures: {X.shape}")
    print(f"Target: mean={y.mean():.2f}, std={y.std():.2f}")
