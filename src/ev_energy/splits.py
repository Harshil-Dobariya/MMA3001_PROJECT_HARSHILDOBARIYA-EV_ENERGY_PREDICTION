"""Leakage-safe train/test splits for the EV dataset.

Many vehicle models appear in several model years with identical
specifications. A random row-wise split would put copies of the same car in
both training and test sets, so the test score would partly measure memory
rather than prediction. Every split here therefore keeps all rows of one
``group_id`` (one make and model) on the same side.

Two splits are provided:

* :func:`group_holdout_split`: the main split. A fixed fraction of vehicle
  models is held out at random. Answers "how well does the method predict a
  vehicle model it has never seen?"
* :func:`group_kfold_indices`: grouped cross-validation folds assigned
  explicitly, so they are identical on every computer and library version.
* :func:`temporal_split`: a secondary check. Trains on earlier model years
  and tests on later-year models never seen in training. Answers "how well
  does the method predict next year's new cars?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

#: Default fraction of vehicle models held out for testing.
TEST_FRACTION = 0.2

#: Seed used for every random choice in the project, for reproducibility.
RANDOM_STATE = 42


def group_holdout_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
    group_col: str = "group_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by vehicle model so no model appears in both sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned vehicle table with a group column.
    test_fraction : float, optional
        Fraction of *groups* (vehicle models) to hold out, between 0 and 1.
    random_state : int, optional
        Seed, so the same split is produced every run.
    group_col : str, optional
        Name of the group column.

    Returns
    -------
    train, test : pandas.DataFrame
        Row subsets of ``df`` with their original index reset.

    Raises
    ------
    ValueError
        If ``test_fraction`` is not strictly between 0 and 1, or the group
        column is missing.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not found.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def group_kfold_indices(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = RANDOM_STATE,
    group_col: str = "group_id",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign whole vehicle models to cross-validation folds, reproducibly.

    scikit-learn's ``GroupKFold`` can assign groups to folds differently in
    different library versions, which changed cross-validation results
    between computers. This function makes the assignment explicit: the
    unique group names are sorted, shuffled with NumPy's legacy
    ``RandomState`` (whose random stream is guaranteed not to change between
    NumPy versions), then dealt to folds in turn like cards.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a group column. Row positions (not index labels) are
        returned.
    n_folds : int, optional
        Number of folds, at least 2 and at most the number of groups.
    random_state : int, optional
        Seed for the shuffle.
    group_col : str, optional
        Name of the group column.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        ``(train_positions, validation_positions)`` for each fold, in the
        format scikit-learn accepts as ``cv=``.

    Raises
    ------
    ValueError
        If ``n_folds`` is less than 2 or more than the number of groups.

    Examples
    --------
    >>> toy = pd.DataFrame({"group_id": ["a", "a", "b", "c", "d"]})
    >>> folds = group_kfold_indices(toy, n_folds=2)
    >>> sorted(len(va) for _, va in folds)
    [2, 3]
    """
    groups = df[group_col].to_numpy()
    unique = np.array(sorted(set(groups)))
    if not 2 <= n_folds <= len(unique):
        raise ValueError(f"n_folds must be between 2 and {len(unique)}.")
    shuffled = np.random.RandomState(random_state).permutation(unique)
    fold_of_group = {g: i % n_folds for i, g in enumerate(shuffled)}
    fold_of_row = np.array([fold_of_group[g] for g in groups])
    positions = np.arange(len(df))
    return [
        (positions[fold_of_row != k], positions[fold_of_row == k]) for k in range(n_folds)
    ]


def temporal_split(
    df: pd.DataFrame,
    cutoff_year: int = 2025,
    year_col: str = "model_year",
    group_col: str = "group_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on years before ``cutoff_year``; test on new models from then on.

    Test rows are those from ``cutoff_year`` onwards whose vehicle model never
    appears in the training years. Later-year rows of models already seen in
    training are discarded, since they are not new cars.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned vehicle table.
    cutoff_year : int, optional
        First model year that belongs to the test period.
    year_col, group_col : str, optional
        Column names.

    Returns
    -------
    train, test : pandas.DataFrame
        Row subsets with the index reset.
    """
    train = df[df[year_col] < cutoff_year]
    later = df[df[year_col] >= cutoff_year]
    test = later[~later[group_col].isin(set(train[group_col]))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def check_no_group_overlap(train: pd.DataFrame, test: pd.DataFrame, group_col: str = "group_id") -> None:
    """Raise an error if any vehicle model appears in both sets.

    Parameters
    ----------
    train, test : pandas.DataFrame
        The two sets to compare.
    group_col : str, optional
        Name of the group column.

    Raises
    ------
    AssertionError
        If any group is shared.
    """
    shared = set(train[group_col]) & set(test[group_col])
    if shared:
        raise AssertionError(f"{len(shared)} vehicle models appear in both sets, e.g. {sorted(shared)[:3]}")


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Summarise the sizes and target spread of a split.

    Parameters
    ----------
    train, test : pandas.DataFrame
        The two sets.

    Returns
    -------
    pandas.DataFrame
        One row per set with row count, model count and the mean and standard
        deviation of both targets.
    """
    rows = []
    for name, part in (("train", train), ("test", test)):
        rows.append(
            {
                "set": name,
                "vehicles": len(part),
                "models": part["group_id"].nunique(),
                "city_mean": part["city_kwh_per_100mi"].mean(),
                "city_std": part["city_kwh_per_100mi"].std(),
                "hwy_mean": part["hwy_kwh_per_100mi"].mean(),
                "hwy_std": part["hwy_kwh_per_100mi"].std(),
            }
        )
    return pd.DataFrame(rows).round(2)
