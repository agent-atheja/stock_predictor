"""Parquet datalake IO — partitioned, idempotent writes. Keeps the store lean (no
monolithic multi-GB DB; RAM on this host is the binding constraint).

Layout:  <layer>/<dataset>/dt=YYYY-MM-DD/part.parquet   (Hive-style, date-partitioned)
The partition key is `dt` (string) to avoid colliding with the real timestamp `date` column
inside each file. Reads use pyarrow dataset scans so we never load more than needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import resolve
from core.logging_setup import get_logger

log = get_logger(__name__)


def write_partitioned(df: pd.DataFrame, layer_dir: str, dataset: str, date_col: str = "date") -> int:
    """Write a frame partitioned by date. Overwrites touched partitions only (idempotent)."""
    if df.empty:
        log.warning("write_partitioned: empty frame for %s/%s — skipping", layer_dir, dataset)
        return 0
    base = resolve(layer_dir) / dataset
    base.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    n = 0
    for d, part in df.groupby(dates):
        pdir = base / f"dt={d}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.to_parquet(pdir / "part.parquet", index=False)
        n += len(part)
    log.info("wrote %d rows to %s/%s across %d partitions", n, layer_dir, dataset, dates.nunique())
    return n


def read_dataset(layer_dir: str, dataset: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read a full dataset (optionally date-bounded) via pyarrow dataset scan."""
    import pyarrow.dataset as ds

    base = resolve(layer_dir) / dataset
    if not base.exists():
        return pd.DataFrame()
    dataset_obj = ds.dataset(base, format="parquet", partitioning="hive")
    table = dataset_obj.to_table()
    df = table.to_pandas()
    df = df.drop(columns=[c for c in ("dt",) if c in df.columns])  # partition key is redundant
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def dataset_exists(layer_dir: str, dataset: str) -> bool:
    return (resolve(layer_dir) / dataset).exists()
